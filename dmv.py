import simpy
import random
import statistics


class DMVSimulation:
    def __init__(
        self,
        env,
        num_clerks=3,
        num_kiosks=2,
        interarrival_time=(0.5, 1.5),   # heavier traffic
        clerk_service_time=(8, 15),
        kiosk_service_time=(4, 8),
        sim_time=480,
        seed=42
    ):
        self.env = env
        random.seed(seed)

        self.num_clerks = num_clerks
        self.num_kiosks = num_kiosks
        self.interarrival_time = interarrival_time
        self.clerk_service_time = clerk_service_time
        self.kiosk_service_time = kiosk_service_time
        self.sim_time = sim_time

        # Shared server pool = one common line
        self.servers = simpy.Store(env)

        for i in range(num_clerks):
            self.servers.put(("clerk", i))

        for i in range(num_kiosks):
            self.servers.put(("kiosk", i))

        # Metrics
        self.completed_customers = 0
        self.wait_times = []
        self.system_times = []

        # Queue length tracking
        self.num_waiting = 0
        self.queue_length_timeline = [(0, 0)]

        # Utilization tracking
        self.total_clerk_busy_time = 0.0
        self.total_kiosk_busy_time = 0.0

    def sample_interarrival(self):
        return random.uniform(*self.interarrival_time)

    def sample_clerk_service(self):
        return random.uniform(*self.clerk_service_time)

    def sample_kiosk_service(self):
        return random.uniform(*self.kiosk_service_time)

    def record_queue_length(self):
        self.queue_length_timeline.append((self.env.now, self.num_waiting))

    def customer(self, customer_id):
        arrival_time = self.env.now

        # Customer joins shared line
        self.num_waiting += 1
        self.record_queue_length()

        # Wait for first available server from shared pool
        server_type, server_id = yield self.servers.get()

        service_start = self.env.now

        # Customer leaves queue and begins service
        self.num_waiting -= 1
        self.record_queue_length()

        wait_time = service_start - arrival_time
        self.wait_times.append(wait_time)

        # Service time depends on server type
        if server_type == "clerk":
            service_time = self.sample_clerk_service()
            self.total_clerk_busy_time += service_time
        else:
            service_time = self.sample_kiosk_service()
            self.total_kiosk_busy_time += service_time

        yield self.env.timeout(service_time)

        # Server becomes available again
        yield self.servers.put((server_type, server_id))

        departure_time = self.env.now
        system_time = departure_time - arrival_time
        self.system_times.append(system_time)

        self.completed_customers += 1

    def arrival_generator(self):
        customer_id = 0
        while True:
            yield self.env.timeout(self.sample_interarrival())
            customer_id += 1
            self.env.process(self.customer(customer_id))

    def time_weighted_average_queue_length(self):
        area = 0.0

        for i in range(len(self.queue_length_timeline) - 1):
            t0, q0 = self.queue_length_timeline[i]
            t1, _ = self.queue_length_timeline[i + 1]
            area += q0 * (t1 - t0)

        last_time, last_q = self.queue_length_timeline[-1]
        if last_time < self.sim_time:
            area += last_q * (self.sim_time - last_time)

        return area / self.sim_time if self.sim_time > 0 else 0.0

    def results(self):
        clerk_capacity_time = self.num_clerks * self.sim_time
        kiosk_capacity_time = self.num_kiosks * self.sim_time
        total_capacity_time = (self.num_clerks + self.num_kiosks) * self.sim_time

        clerk_util = (
            self.total_clerk_busy_time / clerk_capacity_time
            if clerk_capacity_time > 0 else 0.0
        )
        kiosk_util = (
            self.total_kiosk_busy_time / kiosk_capacity_time
            if kiosk_capacity_time > 0 else 0.0
        )
        combined_util = (
            (self.total_clerk_busy_time + self.total_kiosk_busy_time) / total_capacity_time
            if total_capacity_time > 0 else 0.0
        )

        return {
            "completed_customers": self.completed_customers,
            "avg_wait_time": statistics.mean(self.wait_times) if self.wait_times else 0.0,
            "max_wait_time": max(self.wait_times) if self.wait_times else 0.0,
            "avg_time_in_system": statistics.mean(self.system_times) if self.system_times else 0.0,
            "max_time_in_system": max(self.system_times) if self.system_times else 0.0,
            "avg_queue_length": self.time_weighted_average_queue_length(),
            "max_queue_length": max(q for _, q in self.queue_length_timeline),
            "clerk_utilization": clerk_util,
            "kiosk_utilization": kiosk_util,
            "combined_utilization": combined_util,
        }


def run_simulation():
    SIM_TIME = 480  # 8 hours

    env = simpy.Environment()

    sim = DMVSimulation(
        env,
        num_clerks=3,
        num_kiosks=2,
        interarrival_time=(0.5, 1.5),   # heavier arrivals
        clerk_service_time=(8, 15),
        kiosk_service_time=(4, 8),
        sim_time=SIM_TIME,
        seed=42
    )

    env.process(sim.arrival_generator())
    env.run(until=SIM_TIME)

    results = sim.results()

    print("\nDMV Simulation Results")
    print("-" * 35)
    print(f"Completed customers:      {results['completed_customers']}")
    print(f"Average queue length:     {results['avg_queue_length']:.2f}")
    print(f"Maximum queue length:     {results['max_queue_length']}")
    print(f"Average wait time:        {results['avg_wait_time']:.2f} minutes")
    print(f"Maximum wait time:        {results['max_wait_time']:.2f} minutes")
    print(f"Average time in system:   {results['avg_time_in_system']:.2f} minutes")
    print(f"Maximum time in system:   {results['max_time_in_system']:.2f} minutes")
    print(f"Clerk utilization:        {results['clerk_utilization']:.2%}")
    print(f"Kiosk utilization:        {results['kiosk_utilization']:.2%}")
    print(f"Combined utilization:     {results['combined_utilization']:.2%}")


if __name__ == "__main__":
    run_simulation()
