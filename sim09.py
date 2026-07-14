import json
import math
import random

import simpy


# Configuration
NUM_CLERKS = 10
NUM_KIOSKS = 2
USE_KIOSKS = True

SLOT_DURATION = 60
APPS_PER_SLOT = 5
GRACE_PERIOD = 5
SIM_TIME = 420

PRE_OPENING_LINE = 10
APPT_NO_SHOW_RATE = 0.1
WALKIN_PATIENCE = 100

PRINT_EVENTS = False
OUTPUT_FILE = "visitors.json"

APPOINTMENT_CASE_TYPES = {
    "Card_Replacement": {"mean": 5, "probability": 0.15},
    "Retirement_Benefits": {"mean": 15, "probability": 0.45},
    "Disability_Claim": {"mean": 40, "probability": 0.40},
}

WALKIN_CASE_TYPES = {
    "Card_Replacement": {"mean": 5, "probability": 0.70},
    "Retirement_Benefits": {"mean": 15, "probability": 0.20},
    "Disability_Claim": {"mean": 40, "probability": 0.10},
}


def choose_case_type(case_types):
    names = list(case_types)
    probabilities = [case_types[name]["probability"] for name in names]
    return random.choices(names, weights=probabilities, k=1)[0]


def service_duration(case_type, resource_type):
    case_config = CASE_TYPE_LOOKUP[case_type]
    mean = case_config["mean"]

    if resource_type == "kiosk":
        mean *= 0.5

    return max(2, random.expovariate(1 / mean))


def walkin_rate(minute):
    """Return walk-in arrival intensity as expected arrivals per minute."""
    peak_time = SIM_TIME / 2
    peak_rate = 0.15
    stddev = SIM_TIME / 5

    morning_peak_time = 0
    morning_peak_rate = 0.15
    morning_stddev = 25
    morning = morning_peak_rate * math.exp(
        -((minute - morning_peak_time) ** 2) / (2 * morning_stddev**2)
    )

    midday = peak_rate * math.exp(-((minute - peak_time) ** 2) / (2 * stddev**2))
    base = morning + midday
    fluctuation = random.uniform(-0.1, 0.1) * base
    return max(0, base + fluctuation)


def cancel_or_release(resource, request):
    if request is None:
        return

    if request.triggered:
        resource.release(request)
    else:
        request.cancel()


class FieldOffice:
    def __init__(self, env):
        self.env = env
        self.clerks = simpy.PriorityResource(env, capacity=NUM_CLERKS)
        self.kiosks = simpy.Resource(env, capacity=NUM_KIOSKS)
        self.served_apps = 0
        self.served_walkins = 0
        self.walkins_left = 0

    def service_process(self, visitor_type, resource_type, case_type):
        duration = service_duration(case_type, resource_type)
        if PRINT_EVENTS:
            print(
                f"{visitor_type} {case_type} started {resource_type} service "
                f"for {duration:.1f} minutes at {self.env.now:.1f}"
            )
        yield self.env.timeout(duration)


CASE_TYPE_LOOKUP = {
    **APPOINTMENT_CASE_TYPES,
    **WALKIN_CASE_TYPES,
}

stats = {"visitors": []}
visitor_sequence = 0
walkin_total = 0


def next_visitor_id():
    global visitor_sequence
    visitor_sequence += 1
    return visitor_sequence


def queue_snapshot(fo):
    return {
        "clerk_queue_length": len(fo.clerks.queue),
        "clerk_count": fo.clerks.count,
        "kiosk_queue_length": len(fo.kiosks.queue),
        "kiosk_count": fo.kiosks.count,
    }


def appointment_customer(env, fo, slot_start, appointment_id):
    case_type = choose_case_type(APPOINTMENT_CASE_TYPES)
    arrival_time = max(0, slot_start + random.normalvariate(0, 4))
    yield env.timeout(max(0, arrival_time - env.now))

    priority = 0 if env.now <= slot_start + GRACE_PERIOD else 1
    queue_enter_time = env.now
    snapshot = queue_snapshot(fo)

    with fo.clerks.request(priority=priority) as req:
        yield req
        service_start = env.now
        yield env.process(fo.service_process("Appointment", "clerk", case_type))
        departure_time = env.now
        fo.served_apps += 1

    stats["visitors"].append(
        {
            "id": next_visitor_id(),
            "name": f"a-{appointment_id}",
            "type": "appointment",
            "case_type": case_type,
            "resource": "clerk",
            "slot": slot_start,
            "priority": priority,
            "arrival_time": queue_enter_time,
            "service_start": service_start,
            "wait_time": service_start - queue_enter_time,
            "service_time": departure_time - service_start,
            "time_in_system": departure_time - queue_enter_time,
            "departure": departure_time,
            **snapshot,
        }
    )


def serve_walkin(env, fo, walkin_id, req_clerk, req_kiosk, resource_type, case_type, record_base):
    if resource_type == "clerk":
        cancel_or_release(fo.kiosks, req_kiosk)
    else:
        cancel_or_release(fo.clerks, req_clerk)

    service_start = env.now
    yield env.process(fo.service_process("Walk-in", resource_type, case_type))
    departure_time = env.now
    fo.served_walkins += 1

    if resource_type == "clerk":
        fo.clerks.release(req_clerk)
    else:
        fo.kiosks.release(req_kiosk)

    stats["visitors"].append(
        {
            "id": next_visitor_id(),
            "name": f"w-{walkin_id}",
            "type": "walkin",
            "case_type": case_type,
            "resource": resource_type,
            "slot": "",
            "priority": 2,
            "service_start": service_start,
            "wait_time": service_start - record_base["arrival_time"],
            "service_time": departure_time - service_start,
            "time_in_system": departure_time - record_base["arrival_time"],
            "departure": departure_time,
            **record_base,
        }
    )


def walkin_customer(env, fo, walkin_id):
    case_type = choose_case_type(WALKIN_CASE_TYPES)
    req_clerk = fo.clerks.request(priority=2)
    req_kiosk = fo.kiosks.request() if USE_KIOSKS else None
    patience_timer = env.timeout(WALKIN_PATIENCE)

    queue_enter_time = env.now
    record_base = {
        "arrival_time": queue_enter_time,
        **queue_snapshot(fo),
    }

    requests = [req_clerk, patience_timer]
    if req_kiosk is not None:
        requests.append(req_kiosk)

    result = yield env.any_of(requests)

    if req_clerk in result and req_kiosk is not None and req_kiosk in result:
        resource_type = random.choice(["clerk", "kiosk"])
        yield env.process(
            serve_walkin(
                env,
                fo,
                walkin_id,
                req_clerk,
                req_kiosk,
                resource_type,
                case_type,
                record_base,
            )
        )
    elif req_clerk in result:
        yield env.process(
            serve_walkin(
                env,
                fo,
                walkin_id,
                req_clerk,
                req_kiosk,
                "clerk",
                case_type,
                record_base,
            )
        )
    elif req_kiosk is not None and req_kiosk in result:
        yield env.process(
            serve_walkin(
                env,
                fo,
                walkin_id,
                req_clerk,
                req_kiosk,
                "kiosk",
                case_type,
                record_base,
            )
        )
    else:
        cancel_or_release(fo.clerks, req_clerk)
        cancel_or_release(fo.kiosks, req_kiosk)
        fo.walkins_left += 1


def appointment_generator(env, fo):
    appointment_id = 0
    while env.now < SIM_TIME:
        slot_start = env.now
        for _ in range(APPS_PER_SLOT):
            if random.random() < APPT_NO_SHOW_RATE:
                continue

            appointment_id += 1
            env.process(appointment_customer(env, fo, slot_start, appointment_id))

        yield env.timeout(SLOT_DURATION)


def walkin_generator(env, fo):
    global walkin_total
    walkin_id = 0

    for _ in range(random.randint(PRE_OPENING_LINE - 2, PRE_OPENING_LINE + 2)):
        walkin_total += 1
        walkin_id += 1
        env.process(walkin_customer(env, fo, walkin_id))
        yield env.timeout(max(0, random.normalvariate(0.5, 0.1)))

    while env.now < SIM_TIME:
        rate = walkin_rate(env.now)
        delay = random.expovariate(rate) if rate > 0 else 1
        yield env.timeout(delay)

        if env.now >= SIM_TIME:
            break

        walkin_total += 1
        walkin_id += 1
        env.process(walkin_customer(env, fo, walkin_id))


def average_wait(visitor_type):
    waits = [
        visitor["wait_time"]
        for visitor in stats["visitors"]
        if visitor["type"] == visitor_type
    ]
    return sum(waits) / len(waits) if waits else 0


def service_counts_by_case():
    counts = {}
    for visitor in stats["visitors"]:
        case_type = visitor["case_type"]
        counts[case_type] = counts.get(case_type, 0) + 1
    return counts


def main():
    random.seed()
    env = simpy.Environment()
    field_office = FieldOffice(env)

    env.process(appointment_generator(env, field_office))
    env.process(walkin_generator(env, field_office))
    env.run(until=SIM_TIME + 30)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("--- End of Day Report ---")
    print(f"Appointments served: {field_office.served_apps}")
    print(f"Walk-ins total: {walkin_total}")
    print(f"Walk-ins served: {field_office.served_walkins}")
    print(f"Walk-ins left without service: {field_office.walkins_left}")
    print(f"Appointment average wait time: {average_wait('appointment'):.2f}")
    print(f"Walk-in average wait time: {average_wait('walkin'):.2f}")
    print("Served cases by type:")
    for case_type, count in sorted(service_counts_by_case().items()):
        print(f"  {case_type}: {count}")


if __name__ == "__main__":
    main()
