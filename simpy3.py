import simpy
import random
import statistics


# ----------------------------
# Configuration
# ----------------------------
RANDOM_SEED = 42
NUM_CLERKS = 10
OFFICE_MINUTES = 420          # 9:00 AM to 4:00 PM
SLOT_LENGTH = 30              # appointment / walk-in generation interval
APPTS_PER_SLOT = 5
WALKIN_MIN_PER_SLOT = 4
WALKIN_MAX_PER_SLOT = 6
APPT_ARRIVAL_JITTER = 5       # +/- 5 minutes
SERVICE_MIN = 15
SERVICE_MAX = 25

# Priority: smaller number = higher priority in SimPy
PRIORITY_APPT = 0
PRIORITY_WALKIN = 1


# ----------------------------
# Helper functions
# ----------------------------
def service_time():
    return random.uniform(SERVICE_MIN, SERVICE_MAX)


def clamp(x, low, high):
    return max(low, min(high, x))


def minutes_to_clock(minute_from_open):
    """
    Convert minutes since 9:00 AM into HH:MM format.
    """
    total_minutes = 9 * 60 + minute_from_open
    hh = int(total_minutes // 60)
    mm = int(total_minutes % 60)
    return f"{hh:02d}:{mm:02d}"


# ----------------------------
# Visitor process
# ----------------------------
def visitor(env, name, clerks, visitor_type, arrival_time, stats):
    """
    A single DMV visitor.
    """
    yield env.timeout(arrival_time - env.now)

    actual_arrival = env.now
    priority = PRIORITY_APPT if visitor_type == "appointment" else PRIORITY_WALKIN

    with clerks.request(priority=priority) as req:
        queue_enter_time = env.now
        yield req
        service_start = env.now
        wait_time = service_start - queue_enter_time

        st = service_time()
        yield env.timeout(st)
        departure_time = env.now

    # Record stats
    record = {
        "name": name,
        "type": visitor_type,
        "arrival_time": actual_arrival,
        "service_start": service_start,
        "service_time": st,
        "departure_time": departure_time,
        "wait_time": wait_time,
        "time_in_system": departure_time - actual_arrival,
    }
    stats["visitors"].append(record)

    if visitor_type == "appointment":
        stats["appt_waits"].append(wait_time)
        stats["appt_system_times"].append(record["time_in_system"])
    else:
        stats["walkin_waits"].append(wait_time)
        stats["walkin_system_times"].append(record["time_in_system"])


# ----------------------------
# Appointment generator
# ----------------------------
def appointment_generator(env, clerks, stats):
    """
    Generate 5 appointment visitors every 30 minutes.
    Scheduled times are at each slot boundary (0, 30, 60, ...),
    and each appointment visitor arrives around that time +/- 5 minutes.
    """
    visitor_id = 0

    for slot_start in range(0, OFFICE_MINUTES, SLOT_LENGTH):
        scheduled_time = slot_start

        for _ in range(APPTS_PER_SLOT):
            arrival_time = clamp(
                scheduled_time + random.uniform(-APPT_ARRIVAL_JITTER, APPT_ARRIVAL_JITTER),
                0,
                OFFICE_MINUTES
            )
            visitor_id += 1
            env.process(
                visitor(
                    env,
                    name=f"A{visitor_id}",
                    clerks=clerks,
                    visitor_type="appointment",
                    arrival_time=arrival_time,
                    stats=stats
                )
            )

    yield env.timeout(0)


# ----------------------------
# Walk-in generator
# ----------------------------
def walkin_generator(env, clerks, stats):
    """
    Generate 4 to 6 walk-ins every 30 minutes.
    Their arrivals are spread uniformly within that 30-minute slot.
    """
    visitor_id = 0

    for slot_start in range(0, OFFICE_MINUTES, SLOT_LENGTH):
        num_walkins = random.randint(WALKIN_MIN_PER_SLOT, WALKIN_MAX_PER_SLOT)

        for _ in range(num_walkins):
            arrival_time = slot_start + random.uniform(0, SLOT_LENGTH)
            arrival_time = clamp(arrival_time, 0, OFFICE_MINUTES)

            visitor_id += 1
            env.process(
                visitor(
                    env,
                    name=f"W{visitor_id}",
                    clerks=clerks,
                    visitor_type="walkin",
                    arrival_time=arrival_time,
                    stats=stats
                )
            )

    yield env.timeout(0)


# ----------------------------
# Main simulation runner
# ----------------------------
def run_simulation(seed=RANDOM_SEED, verbose=False):
    random.seed(seed)
    env = simpy.Environment()

    # PriorityResource so appointments get served before walk-ins
    clerks = simpy.PriorityResource(env, capacity=NUM_CLERKS)

    stats = {
        "visitors": [],
        "appt_waits": [],
        "walkin_waits": [],
        "appt_system_times": [],
        "walkin_system_times": [],
    }

    env.process(appointment_generator(env, clerks, stats))
    env.process(walkin_generator(env, clerks, stats))

    # Run long enough so everyone who arrived before closing can finish
    env.run(until=OFFICE_MINUTES + 300)

    if verbose:
        for rec in sorted(stats["visitors"], key=lambda x: x["arrival_time"]):
            print(
                f"{rec['name']:>4} | {rec['type']:<11} | "
                f"arrive={minutes_to_clock(rec['arrival_time'])} | "
                f"start={minutes_to_clock(rec['service_start'])} | "
                f"wait={rec['wait_time']:.1f} | "
                f"svc={rec['service_time']:.1f} | "
                f"leave={minutes_to_clock(rec['departure_time'])}"
            )

    return stats


# ----------------------------
# Reporting
# ----------------------------
def summarize(stats):
    total = len(stats["visitors"])
    appt_n = len(stats["appt_waits"])
    walkin_n = len(stats["walkin_waits"])

    def safe_mean(x):
        return statistics.mean(x) if x else 0

    def safe_median(x):
        return statistics.median(x) if x else 0

    def safe_max(x):
        return max(x) if x else 0

    print("\n===== DMV Simulation Summary =====")
    print(f"Total visitors served: {total}")
    print(f"Appointment visitors: {appt_n}")
    print(f"Walk-in visitors:     {walkin_n}")

    print("\n--- Average Wait Time (minutes) ---")
    print(f"Appointments: {safe_mean(stats['appt_waits']):.2f}")
    print(f"Walk-ins:     {safe_mean(stats['walkin_waits']):.2f}")
    print(f"Overall:      {safe_mean(stats['appt_waits'] + stats['walkin_waits']):.2f}")

    print("\n--- Median Wait Time (minutes) ---")
    print(f"Appointments: {safe_median(stats['appt_waits']):.2f}")
    print(f"Walk-ins:     {safe_median(stats['walkin_waits']):.2f}")

    print("\n--- Max Wait Time (minutes) ---")
    print(f"Appointments: {safe_max(stats['appt_waits']):.2f}")
    print(f"Walk-ins:     {safe_max(stats['walkin_waits']):.2f}")

    print("\n--- Average Time in System (minutes) ---")
    print(f"Appointments: {safe_mean(stats['appt_system_times']):.2f}")
    print(f"Walk-ins:     {safe_mean(stats['walkin_system_times']):.2f}")
    print(f"Overall:      {safe_mean(stats['appt_system_times'] + stats['walkin_system_times']):.2f}")


# ----------------------------
# Multiple replications
# ----------------------------
def run_multiple_replications(n_runs=100):
    avg_appt_waits = []
    avg_walkin_waits = []
    avg_overall_waits = []

    for i in range(n_runs):
        stats = run_simulation(seed=RANDOM_SEED + i, verbose=False)
        appt_avg = statistics.mean(stats["appt_waits"]) if stats["appt_waits"] else 0
        walkin_avg = statistics.mean(stats["walkin_waits"]) if stats["walkin_waits"] else 0
        overall_avg = statistics.mean(stats["appt_waits"] + stats["walkin_waits"]) if (stats["appt_waits"] or stats["walkin_waits"]) else 0

        avg_appt_waits.append(appt_avg)
        avg_walkin_waits.append(walkin_avg)
        avg_overall_waits.append(overall_avg)

    print("\n===== Replication Results =====")
    print(f"Runs: {n_runs}")
    print(f"Mean appointment wait: {statistics.mean(avg_appt_waits):.2f} minutes")
    print(f"Mean walk-in wait:     {statistics.mean(avg_walkin_waits):.2f} minutes")
    print(f"Mean overall wait:     {statistics.mean(avg_overall_waits):.2f} minutes")


if __name__ == "__main__":
    # Single run
    stats = run_simulation(verbose=False)
    summarize(stats)

    # Optional: many runs for more stable averages
    run_multiple_replications(n_runs=100)
