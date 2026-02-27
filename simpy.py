import simpy
import random
import numpy as np

# -----------------------------
# Utilities: metrics
# -----------------------------
def summarize_waits(waits, label):
    if len(waits) == 0:
        return {f"{label}_n": 0}
    arr = np.array(waits, dtype=float)
    return {
        f"{label}_n": len(arr),
        f"{label}_avg_wait_min": float(arr.mean()),
        f"{label}_p50_wait_min": float(np.percentile(arr, 50)),
        f"{label}_p95_wait_min": float(np.percentile(arr, 95)),
        f"{label}_p99_wait_min": float(np.percentile(arr, 99)),
    }

# -----------------------------
# Core simulation
# -----------------------------
def customer(env, name, kind, clerks, priority, service_time_sampler, waits_out):
    """
    kind: 'appt' or 'walkin'
    priority: lower number => higher priority in SimPy PriorityResource
    """
    arrive = env.now
    with clerks.request(priority=priority) as req:
        yield req
        start = env.now
        waits_out[kind].append(start - arrive)  # minutes
        service_time = service_time_sampler(kind)
        yield env.timeout(service_time)

def arrival_process(env, kind, clerks, priority, interarrival_sampler, service_time_sampler, waits_out, until):
    i = 0
    while env.now < until:
        i += 1
        env.process(customer(
            env=env,
            name=f"{kind}_{i}",
            kind=kind,
            clerks=clerks,
            priority=priority,
            service_time_sampler=service_time_sampler,
            waits_out=waits_out
        ))
        yield env.timeout(interarrival_sampler(kind))

# -----------------------------
# Example parameterization
# -----------------------------
def run_dmv_sim(
    seed=7,
    sim_minutes=8*60,       # simulate 1 workday (8 hours)
    num_clerks=6,

    # Baseline arrival rates (customers per hour)
    appt_per_hour=60,
    walkin_per_hour=30,

    # Policy shock: walk-ins increase by 20%
    walkin_multiplier=1.2,

    # Service time assumptions (minutes) by class (can be same too)
    mean_service_appt=6.0,
    mean_service_walkin=6.0,

    # Priority rule: appointments first
    appt_priority=0,
    walkin_priority=1,
):
    random.seed(seed)
    np.random.seed(seed)

    # Convert rates to mean interarrival times (minutes)
    appt_rate_per_min = appt_per_hour / 60.0
    walkin_rate_per_min = (walkin_per_hour * walkin_multiplier) / 60.0

    def interarrival_sampler(kind):
        # Poisson arrivals => exponential interarrival times
        if kind == "appt":
            rate = appt_rate_per_min
        else:
            rate = walkin_rate_per_min
        # Guard against rate=0
        return float("inf") if rate <= 0 else random.expovariate(rate)

    def service_time_sampler(kind):
        # Exponential service times (replace with lognormal / empirical later)
        mean = mean_service_appt if kind == "appt" else mean_service_walkin
        return random.expovariate(1.0 / mean)  # mean minutes

    env = simpy.Environment()
    clerks = simpy.PriorityResource(env, capacity=num_clerks)

    waits_out = {"appt": [], "walkin": []}

    env.process(arrival_process(env, "appt", clerks, appt_priority, interarrival_sampler, service_time_sampler, waits_out, sim_minutes))
    env.process(arrival_process(env, "walkin", clerks, walkin_priority, interarrival_sampler, service_time_sampler, waits_out, sim_minutes))

    env.run(until=sim_minutes)

    results = {}
    results.update(summarize_waits(waits_out["appt"], "appt"))
    results.update(summarize_waits(waits_out["walkin"], "walkin"))
    return results

# -----------------------------
# Run baseline vs new policy
# -----------------------------
if __name__ == "__main__":
    # Baseline (no increase)
    base = run_dmv_sim(walkin_multiplier=1.0)
    # Policy (+20% walk-ins)
    newp = run_dmv_sim(walkin_multiplier=1.2)

    print("Baseline:", base)
    print("Policy +20% walk-ins:", newp)
