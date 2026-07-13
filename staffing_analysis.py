import argparse
import math
import random
from statistics import mean

import simpy


SIM_TIME = 420
SLOT_DURATION = 60
APPS_PER_SLOT = 5
GRACE_PERIOD = 5
NUM_KIOSKS = 2
USE_KIOSKS = True
PRE_OPENING_LINE = 10
APPT_NO_SHOW_RATE = 0.1
WALKIN_PATIENCE = 100

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

CASE_TYPE_LOOKUP = {
    **APPOINTMENT_CASE_TYPES,
    **WALKIN_CASE_TYPES,
}

DEFAULT_WAIT_GOAL = {
    "appointment_avg_wait": 10,
    "appointment_p90_wait": 20,
    "walkin_avg_wait": 30,
    "walkin_p90_wait": 60,
    "overall_p90_wait": 45,
    "walkin_abandonment_rate": 0.10,
}


def choose_case_type(case_types):
    names = list(case_types)
    probabilities = [case_types[name]["probability"] for name in names]
    return random.choices(names, weights=probabilities, k=1)[0]


def service_duration(case_type, resource_type):
    mean_minutes = CASE_TYPE_LOOKUP[case_type]["mean"]
    if resource_type == "kiosk":
        mean_minutes *= 0.5

    return max(2, random.expovariate(1 / mean_minutes))


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


def percentile(values, pct):
    if not values:
        return 0

    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return ordered[int(index)]

    lower_weight = upper - index
    upper_weight = index - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


def cancel_or_release(resource, request):
    if request is None:
        return

    if request.triggered:
        resource.release(request)
    else:
        request.cancel()


class FieldOffice:
    def __init__(self, env, num_clerks, num_kiosks):
        self.env = env
        self.num_clerks = num_clerks
        self.num_kiosks = num_kiosks
        self.clerks = simpy.PriorityResource(env, capacity=num_clerks)
        self.kiosks = simpy.Resource(env, capacity=num_kiosks)
        self.served_apps = 0
        self.served_walkins = 0
        self.walkins_left = 0
        self.walkins_total = 0
        self.clerk_busy_minutes = 0
        self.kiosk_busy_minutes = 0
        self.last_departure = 0

    def service_process(self, resource_type, case_type):
        duration = service_duration(case_type, resource_type)
        yield self.env.timeout(duration)

        if resource_type == "clerk":
            self.clerk_busy_minutes += duration
        else:
            self.kiosk_busy_minutes += duration

        self.last_departure = max(self.last_departure, self.env.now)


def appointment_customer(env, fo, stats, slot_start, appointment_id):
    case_type = choose_case_type(APPOINTMENT_CASE_TYPES)
    arrival_time = max(0, slot_start + random.normalvariate(0, 4))
    yield env.timeout(max(0, arrival_time - env.now))

    priority = 0 if env.now <= slot_start + GRACE_PERIOD else 1
    queue_enter_time = env.now

    with fo.clerks.request(priority=priority) as req:
        yield req
        service_start = env.now
        yield env.process(fo.service_process("clerk", case_type))
        departure_time = env.now
        fo.served_apps += 1

    stats.append(
        {
            "type": "appointment",
            "name": f"a-{appointment_id}",
            "case_type": case_type,
            "resource": "clerk",
            "arrival_time": queue_enter_time,
            "service_start": service_start,
            "wait_time": service_start - queue_enter_time,
            "service_time": departure_time - service_start,
            "departure": departure_time,
        }
    )


def serve_walkin(env, fo, stats, walkin_id, resource_type, case_type, record_base):
    service_start = env.now
    yield env.process(fo.service_process(resource_type, case_type))
    departure_time = env.now
    fo.served_walkins += 1

    stats.append(
        {
            "type": "walkin",
            "name": f"w-{walkin_id}",
            "case_type": case_type,
            "resource": resource_type,
            "service_start": service_start,
            "wait_time": service_start - record_base["arrival_time"],
            "service_time": departure_time - service_start,
            "departure": departure_time,
            **record_base,
        }
    )


def walkin_customer(env, fo, stats, walkin_id, use_kiosks):
    case_type = choose_case_type(WALKIN_CASE_TYPES)
    req_clerk = fo.clerks.request(priority=2)
    req_kiosk = fo.kiosks.request() if use_kiosks else None
    patience_timer = env.timeout(WALKIN_PATIENCE)
    record_base = {"arrival_time": env.now}

    requests = [req_clerk, patience_timer]
    if req_kiosk is not None:
        requests.append(req_kiosk)

    result = yield env.any_of(requests)

    if req_clerk in result and req_kiosk is not None and req_kiosk in result:
        resource_type = random.choice(["clerk", "kiosk"])
        if resource_type == "clerk":
            fo.kiosks.release(req_kiosk)
        else:
            fo.clerks.release(req_clerk)

        yield env.process(
            serve_walkin(env, fo, stats, walkin_id, resource_type, case_type, record_base)
        )

        if resource_type == "clerk":
            fo.clerks.release(req_clerk)
        else:
            fo.kiosks.release(req_kiosk)
    elif req_clerk in result:
        cancel_or_release(fo.kiosks, req_kiosk)
        yield env.process(
            serve_walkin(env, fo, stats, walkin_id, "clerk", case_type, record_base)
        )
        fo.clerks.release(req_clerk)
    elif req_kiosk is not None and req_kiosk in result:
        cancel_or_release(fo.clerks, req_clerk)
        yield env.process(
            serve_walkin(env, fo, stats, walkin_id, "kiosk", case_type, record_base)
        )
        fo.kiosks.release(req_kiosk)
    else:
        cancel_or_release(fo.clerks, req_clerk)
        cancel_or_release(fo.kiosks, req_kiosk)
        fo.walkins_left += 1


def appointment_generator(env, fo, stats):
    appointment_id = 0
    while env.now < SIM_TIME:
        slot_start = env.now
        for _ in range(APPS_PER_SLOT):
            if random.random() < APPT_NO_SHOW_RATE:
                continue

            appointment_id += 1
            env.process(appointment_customer(env, fo, stats, slot_start, appointment_id))

        yield env.timeout(SLOT_DURATION)


def walkin_generator(env, fo, stats, use_kiosks):
    walkin_id = 0

    for _ in range(random.randint(PRE_OPENING_LINE - 2, PRE_OPENING_LINE + 2)):
        fo.walkins_total += 1
        walkin_id += 1
        env.process(walkin_customer(env, fo, stats, walkin_id, use_kiosks))
        yield env.timeout(max(0, random.normalvariate(0.5, 0.1)))

    while env.now < SIM_TIME:
        rate = walkin_rate(env.now)
        delay = random.expovariate(rate) if rate > 0 else 1
        yield env.timeout(delay)

        if env.now >= SIM_TIME:
            break

        fo.walkins_total += 1
        walkin_id += 1
        env.process(walkin_customer(env, fo, stats, walkin_id, use_kiosks))


def summarize_day(fo, visitors):
    appointment_waits = [
        visitor["wait_time"] for visitor in visitors if visitor["type"] == "appointment"
    ]
    walkin_waits = [
        visitor["wait_time"] for visitor in visitors if visitor["type"] == "walkin"
    ]
    all_waits = appointment_waits + walkin_waits
    horizon = max(SIM_TIME, fo.last_departure)

    return {
        "appointments_served": fo.served_apps,
        "walkins_total": fo.walkins_total,
        "walkins_served": fo.served_walkins,
        "walkins_left": fo.walkins_left,
        "appointment_avg_wait": mean(appointment_waits) if appointment_waits else 0,
        "appointment_p90_wait": percentile(appointment_waits, 0.90),
        "walkin_avg_wait": mean(walkin_waits) if walkin_waits else 0,
        "walkin_p90_wait": percentile(walkin_waits, 0.90),
        "overall_avg_wait": mean(all_waits) if all_waits else 0,
        "overall_p90_wait": percentile(all_waits, 0.90),
        "walkin_abandonment_rate": (
            fo.walkins_left / fo.walkins_total if fo.walkins_total else 0
        ),
        "clerk_utilization": fo.clerk_busy_minutes / (fo.num_clerks * horizon),
        "kiosk_utilization": (
            fo.kiosk_busy_minutes / (fo.num_kiosks * horizon) if fo.num_kiosks else 0
        ),
        "last_departure": fo.last_departure,
    }


def run_simulation(num_clerks=10, num_kiosks=NUM_KIOSKS, use_kiosks=USE_KIOSKS, seed=None):
    if seed is not None:
        random.seed(seed)

    env = simpy.Environment()
    visitors = []
    field_office = FieldOffice(env, num_clerks, num_kiosks)

    env.process(appointment_generator(env, field_office, visitors))
    env.process(walkin_generator(env, field_office, visitors, use_kiosks))
    env.run()

    return summarize_day(field_office, visitors)


def aggregate_results(day_results):
    if not day_results:
        return {}

    metric_names = day_results[0].keys()
    return {
        metric: mean(day[metric] for day in day_results)
        for metric in metric_names
    }


def meets_wait_goal(metrics, wait_goal):
    return all(metrics[name] <= target for name, target in wait_goal.items())


def run_replications(num_clerks, replications, seed_start, num_kiosks, use_kiosks):
    return [
        run_simulation(
            num_clerks=num_clerks,
            num_kiosks=num_kiosks,
            use_kiosks=use_kiosks,
            seed=seed_start + i,
        )
        for i in range(replications)
    ]


def find_staffing_for_goal(
    min_clerks=1,
    max_clerks=30,
    replications=100,
    wait_goal=None,
    seed_start=1000,
    num_kiosks=NUM_KIOSKS,
    use_kiosks=USE_KIOSKS,
):
    wait_goal = wait_goal or DEFAULT_WAIT_GOAL
    search_results = []

    for num_clerks in range(min_clerks, max_clerks + 1):
        day_results = run_replications(
            num_clerks=num_clerks,
            replications=replications,
            seed_start=seed_start,
            num_kiosks=num_kiosks,
            use_kiosks=use_kiosks,
        )
        metrics = aggregate_results(day_results)
        passed = meets_wait_goal(metrics, wait_goal)
        search_results.append(
            {
                "num_clerks": num_clerks,
                "passed": passed,
                **metrics,
            }
        )

        if passed:
            return num_clerks, search_results

    return None, search_results


def format_minutes(value):
    return f"{value:.1f} min"


def print_staffing_result(recommended_clerks, search_results, wait_goal, replications):
    print("--- Staffing Search ---")
    print(f"Replications per staffing level: {replications}")
    print("Wait goals:")
    for name, target in wait_goal.items():
        suffix = "" if name.endswith("rate") else " min"
        print(f"  {name}: <= {target}{suffix}")

    print()
    print("Clerks | Pass | Appt avg | Appt p90 | Walk-in avg | Walk-in p90 | Abandon")
    print("-------|------|----------|----------|-------------|-------------|--------")
    for row in search_results:
        print(
            f"{row['num_clerks']:>6} | "
            f"{'yes' if row['passed'] else 'no ':>4} | "
            f"{format_minutes(row['appointment_avg_wait']):>8} | "
            f"{format_minutes(row['appointment_p90_wait']):>8} | "
            f"{format_minutes(row['walkin_avg_wait']):>11} | "
            f"{format_minutes(row['walkin_p90_wait']):>11} | "
            f"{row['walkin_abandonment_rate'] * 100:>5.1f}%"
        )

    print()
    if recommended_clerks is None:
        print("No staffing level in the search range met the wait goals.")
    else:
        best = search_results[-1]
        print(f"Minimum clerks needed: {recommended_clerks}")
        print(
            f"At {recommended_clerks} clerks: "
            f"appointment avg {best['appointment_avg_wait']:.1f} min, "
            f"appointment p90 {best['appointment_p90_wait']:.1f} min, "
            f"walk-in avg {best['walkin_avg_wait']:.1f} min, "
            f"walk-in p90 {best['walkin_p90_wait']:.1f} min, "
            f"abandonment {best['walkin_abandonment_rate'] * 100:.1f}%."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the minimum number of clerks needed to meet wait goals."
    )
    parser.add_argument("--min-clerks", type=int, default=1)
    parser.add_argument("--max-clerks", type=int, default=30)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--num-kiosks", type=int, default=NUM_KIOSKS)
    parser.add_argument("--no-kiosks", action="store_true")
    parser.add_argument("--appointment-avg-wait", type=float, default=10)
    parser.add_argument("--appointment-p90-wait", type=float, default=20)
    parser.add_argument("--walkin-avg-wait", type=float, default=30)
    parser.add_argument("--walkin-p90-wait", type=float, default=60)
    parser.add_argument("--overall-p90-wait", type=float, default=45)
    parser.add_argument("--walkin-abandonment-rate", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    wait_goal = {
        "appointment_avg_wait": args.appointment_avg_wait,
        "appointment_p90_wait": args.appointment_p90_wait,
        "walkin_avg_wait": args.walkin_avg_wait,
        "walkin_p90_wait": args.walkin_p90_wait,
        "overall_p90_wait": args.overall_p90_wait,
        "walkin_abandonment_rate": args.walkin_abandonment_rate,
    }
    recommended_clerks, search_results = find_staffing_for_goal(
        min_clerks=args.min_clerks,
        max_clerks=args.max_clerks,
        replications=args.replications,
        wait_goal=wait_goal,
        seed_start=args.seed_start,
        num_kiosks=args.num_kiosks,
        use_kiosks=not args.no_kiosks,
    )
    print_staffing_result(
        recommended_clerks=recommended_clerks,
        search_results=search_results,
        wait_goal=wait_goal,
        replications=args.replications,
    )


if __name__ == "__main__":
    main()
