import simpy
import random
import numpy as np
from collections import defaultdict

# -----------------------------
# Helpers: distributions
# -----------------------------
def sample_lognormal(mean, sigma, rng=random):
    """
    Lognormal with specified mean (in minutes) and log-space sigma.
    Useful for right-skewed service times.
    """
    # Convert desired mean to mu for lognormal: mean = exp(mu + 0.5*sigma^2)
    mu = np.log(mean) - 0.5 * sigma**2
    return float(rng.lognormvariate(mu, sigma))

def summarize(values):
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=float)
    return {
        "n": int(len(arr)),
        "avg": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }

# -----------------------------
# Core customer process with abandonment
# -----------------------------
def dmv_customer(env, kind, service_type, clerks, priority, patience_min, service_time_sampler, metrics):
    """
    kind: 'appt' or 'walkin'
    service_type: e.g. 'renewal', 'id', 'written_test'
    clerks: shared PriorityResource
    priority: lower = higher priority
    patience_min: maximum time willing to wait before abandoning
    service_time_sampler(service_type, kind) -> minutes
    metrics: dict collecting stats
    """
    arrive = env.now
    metrics["arrivals"][kind] += 1
    metrics["arrivals_by_type"][(kind, service_type)] += 1

    req = clerks.request(priority=priority)
    # Compete "get clerk" vs "patience runs out"
    results = yield req | env.timeout(patience_min)

    if req not in results:
        # Abandoned before being served
        clerks.release(req)  # safe even if not acquired in newer SimPy, but keep explicit
        metrics["abandons"][kind] += 1
        metrics["abandon_waits"][kind].append(env.now - arrive)
        return

    # Got a clerk
    start = env.now
    metrics["waits"][kind].append(start - arrive)
    metrics["waits_by_type"][(kind, service_type)].append(start - arrive)

    service_time = service_time_sampler(service_type, kind)
    yield env.timeout(service_time)

    metrics["services"][kind] += 1
    metrics["service_times"][kind].append(service_time)
    metrics["service_times_by_type"][(kind, service_type)].append(service_time)

    clerks.release(req)

# -----------------------------
# Appointment schedule (15-minute slots)
# -----------------------------
def appointment_schedule_process(
    env,
    clerks,
    slot_minutes,
    slots,                  # list of appointment counts per slot
    service_type_sampler,   # function(kind)-> service_type
    service_time_sampler,
    metrics,
    priority=0,
    patience_min=20.0,
    no_show_rate=0.05
):
    """
    Every slot_minutes, spawn exactly `slots[i]` appointment customers at the slot boundary.
    """
    for slot_idx, n_appts in enumerate(slots):
        # Spawn n_appts at this slot boundary
        for _ in range(n_appts):
            if random.random() < no_show_rate:
                metrics["no_shows"] += 1
                continue
            stype = service_type_sampler("appt")
            env.process(dmv_customer(
                env=env,
                kind="appt",
                service_type=stype,
                clerks=clerks,
                priority=priority,
                patience_min=patience_min,
                service_time_sampler=service_time_sampler,
                metrics=metrics
            ))
        # Move to next slot boundary
        yield env.timeout(slot_minutes)

# -----------------------------
# Walk-in arrivals with per-hour cap
# -----------------------------
def walkin_arrival_process(
    env,
    clerks,
    per_hour_caps,          # list of caps per hour (length = hours)
    hourly_rate,            # expected walk-ins per hour (Poisson-ish)
    service_type_sampler,
    service_time_sampler,
    metrics,
    priority=1,
    patience_min=45.0,
    sim_minutes=8*60
):
    """
    Walk-ins arrive randomly (Poisson), but each hour has a maximum admissions cap.
    If cap is reached, additional arrivals are turned away (counted as rejected).
    """
    # Track admissions within current hour
    current_hour = 0
    admitted_this_hour = 0
    next_hour_boundary = 60.0

    rate_per_min = hourly_rate / 60.0

    while env.now < sim_minutes:
        # If we crossed into next hour, reset counter
        if env.now >= next_hour_boundary:
            current_hour += 1
            admitted_this_hour = 0
            next_hour_boundary += 60.0

        # Sample next arrival time
        if rate_per_min <= 0:
            break
        interarrival = random.expovariate(rate_per_min)
        yield env.timeout(interarrival)

        # Determine hour index safely
        hour_idx = min(int(env.now // 60), len(per_hour_caps) - 1)
        cap = per_hour_caps[hour_idx]

        if admitted_this_hour >= cap:
            metrics["rejected_walkins"] += 1
            continue

        admitted_this_hour += 1
        stype = service_type_sampler("walkin")
        env.process(dmv_customer(
            env=env,
            kind="walkin",
            service_type=stype,
            clerks=clerks,
            priority=priority,
            patience_min=patience_min,
            service_time_sampler=service_time_sampler,
            metrics=metrics
        ))

# -----------------------------
# Main runner
# -----------------------------
def run_dmv_realistic(
    seed=7,
    hours=8,
    num_clerks=6,

    # Appointment schedule: counts per 15-min slot (hours*4 slots)
    slot_minutes=15,
    appt_slots=None,

    # Walk-ins
    walkin_rate_per_hour=30,
    walkin_multiplier=1.2,
    walkin_caps_per_hour=None,

    # Patience / abandonment thresholds
    appt_patience_min=20.0,
    walkin_patience_min=45.0,

    # No-show rate for appointments
    appt_no_show_rate=0.05,
):
    random.seed(seed)
    np.random.seed(seed)

    sim_minutes = hours * 60
    if appt_slots is None:
        # Default: 8 hours -> 32 slots; example: heavier mid-day
        appt_slots = [12]*8 + [16]*16 + [12]*8  # total 32 slots

    if len(appt_slots) != hours * (60 // slot_minutes):
        raise ValueError("appt_slots length must equal hours*(60/slot_minutes).")

    if walkin_caps_per_hour is None:
        # Default: cap by hour (8 values for 8 hours)
        walkin_caps_per_hour = [25, 25, 30, 30, 30, 30, 25, 20]

    if len(walkin_caps_per_hour) != hours:
        raise ValueError("walkin_caps_per_hour length must equal `hours`.")

    # Define service types and mixes (you should calibrate these)
    service_types = ["renewal", "real_id", "knowledge_test", "registration", "other"]

    # Mix by kind (probabilities sum to 1)
    mix_appt = {
        "renewal": 0.35,
        "real_id": 0.30,
        "registration": 0.20,
        "other": 0.15,
        "knowledge_test": 0.00,  # usually walk-in heavy; adjust as needed
    }
    mix_walkin = {
        "renewal": 0.25,
        "real_id": 0.15,
        "knowledge_test": 0.25,
        "registration": 0.15,
        "other": 0.20,
    }

    def service_type_sampler(kind):
        mix = mix_appt if kind == "appt" else mix_walkin
        r = random.random()
        cum = 0.0
        for st, p in mix.items():
            cum += p
            if r <= cum:
                return st
        return "other"

    # Service time assumptions by type (mean minutes + sigma for lognormal)
    service_time_params = {
        "renewal": (6.0, 0.45),
        "real_id": (14.0, 0.55),
        "knowledge_test": (18.0, 0.50),
        "registration": (12.0, 0.55),
        "other": (8.0, 0.60),
    }

    def service_time_sampler(service_type, kind):
        mean, sigma = service_time_params[service_type]
        # Optional: if walk-ins are slightly slower due to prep/doc issues
        if kind == "walkin":
            mean *= 1.05
        return sample_lognormal(mean=mean, sigma=sigma, rng=random)

    env = simpy.Environment()
    clerks = simpy.PriorityResource(env, capacity=num_clerks)

    metrics = {
        "arrivals": defaultdict(int),
        "services": defaultdict(int),
        "abandons": defaultdict(int),
        "waits": defaultdict(list),
        "service_times": defaultdict(list),
        "abandon_waits": defaultdict(list),
        "arrivals_by_type": defaultdict(int),
        "waits_by_type": defaultdict(list),
        "service_times_by_type": defaultdict(list),
        "rejected_walkins": 0,
        "no_shows": 0,
    }

    # Start processes
    env.process(appointment_schedule_process(
        env=env,
        clerks=clerks,
        slot_minutes=slot_minutes,
        slots=appt_slots,
        service_type_sampler=service_type_sampler,
        service_time_sampler=service_time_sampler,
        metrics=metrics,
        priority=0,
        patience_min=appt_patience_min,
        no_show_rate=appt_no_show_rate
    ))

    env.process(walkin_arrival_process(
        env=env,
        clerks=clerks,
        per_hour_caps=walkin_caps_per_hour,
        hourly_rate=walkin_rate_per_hour * walkin_multiplier,
        service_type_sampler=service_type_sampler,
        service_time_sampler=service_time_sampler,
        metrics=metrics,
        priority=1,
        patience_min=walkin_patience_min,
        sim_minutes=sim_minutes
    ))

    env.run(until=sim_minutes)

    # Summaries
    out = {
        "appt_wait": summarize(metrics["waits"]["appt"]),
        "walkin_wait": summarize(metrics["waits"]["walkin"]),
        "appt_service_time": summarize(metrics["service_times"]["appt"]),
        "walkin_service_time": summarize(metrics["service_times"]["walkin"]),
        "arrivals": dict(metrics["arrivals"]),
        "services": dict(metrics["services"]),
        "abandons": dict(metrics["abandons"]),
        "rejected_walkins": metrics["rejected_walkins"],
        "no_shows": metrics["no_shows"],
    }

    return out, metrics

# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    baseline, _ = run_dmv_realistic(walkin_multiplier=1.0)
    policy, _ = run_dmv_realistic(walkin_multiplier=1.2)

    print("Baseline summary:")
    print(baseline)
    print("\nPolicy (+20% walk-ins) summary:")
    print(policy)
