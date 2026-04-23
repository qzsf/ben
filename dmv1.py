import simpy
import random
import statistics


class DMVPrioritySimulation:
    def __init__(
        self,
        env,
        num_clerks=3,
        num_kiosks=2,
        sim_time=480,
        appt_interarrival=(2, 4),
        walkin_interarrival=(0.8, 1.5),
        clerk_service_time=(8, 15),
        kiosk_service_time=(4, 8),
        seed=42
    ):
        self.env = env
        random.seed(seed)

        self.num_clerks = num_clerks
        self.num_kiosks = num_kiosks
        self.sim_time = sim_time

        self.appt_interarrival = appt_interarrival
        self.walkin_interarrival = walkin_interarrival
        self.clerk_service_time = clerk_service_time
        self.kiosk_service_time = kiosk_service_time

        # Two waiting lines
        self.appointment_queue = simpy.Store(env)
        self.walkin_queue = simpy.Store(env)

        # Metrics
        self.completed_customers = 0
        self.completed_appointments = 0
        self.completed_walkins = 0

        self.wait_times_all = []
        self.wait_times_appt = []
        self.wait_times_walkin = []

        self.system_times_all = []
        self.system_times_appt = []
        self.system_times_walkin = []

        self.total_clerk_busy_time = 0.0
        self.total_kiosk_busy_time = 0.0

        self.queue_length_timeline = [(0, 0, 0)]  # (time, appt_q, walkin_q)

        self.customer_id = 0

    def next_customer_id(self):
        self.customer_id += 1
        return self.customer_id

    def sample_appt_interarrival(self):
        return random.uniform(*self.appt_interarrival)

    def sample_walkin_interarrival(self):
        return random.uniform(*self.walkin_interarrival)

    def sample_clerk_service(self):
        return random.uniform(*self.clerk_service_time)

    def sample_kiosk_service(self):
        return random.uniform(*self.kiosk_service_time)

    def record_queue_lengths(self):
        self.queue_length_timeline.append(
            (self.env.now, len(self.appointment_queue.items), len(self.walkin_queue.items))
        )

    def appointment_arrivals(self):
        while True:
            yield self.env.timeout(self.sample_appt_interarrival())
            cid = self.next_customer_id()
            customer = {
                "id": cid,
                "type": "appointment",
                "arrival_time": self.env.now
            }
            yield self.appointment_queue.put(customer)
            self.record_queue_lengths()

    def walkin_arrivals(self):
        while True:
            yield self.env.timeout(self.sample_walkin_interarrival())
            cid = self.next_customer_id()
            customer = {
                "id": cid,
                "type": "walkin",
                "arrival_time": self.env.now
            }
            yield self.walkin_queue.put(customer)
            self.record_queue_lengths()

    def clerk_process(self, clerk_id):
        while True:
            # Clerks prioritize appointments over walk-ins
            if len(self.appointment_queue.items) > 0:
                customer = yield self.appointment_queue.get()
            elif len(self.walkin_queue.items) > 0:
                customer = yield self.walkin_queue.get()
            else:
                # If both queues empty, wait a tiny amount and check again
                yield self.env.timeout(0.1)
                continue

            self.record_queue_lengths()

            service_start = self.env.now
            wait_time = service_start - customer["arrival_time"]

            self.wait_times_all.append(wait_time)
            if customer["type"] == "appointment":
                self.wait_times_appt.append(wait_time)
            else:
                self.wait_times_walkin.append(wait_time)

            service_time = self.sample_clerk_service()
            self.total_clerk_busy_time += service_time

            yield self.env.timeout(service_time)

            departure_time = self.env.now
            system_time = departure_time - customer["arrival_time"]

            self.system_times_all.append(system_time)
            if customer["type"] == "appointment":
                self.system_times_appt.append(system_time)
                self.completed_appointments += 1
            else:
                self.system_times_walkin.append(system_time)
                self.completed_walkins += 1

            self.completed_customers += 1

    def kiosk_process(self, kiosk_id):
        while True:
            # Kiosks serve walk-ins only
            if len(self.walkin_queue.items) > 0:
                customer = yield self.walkin_queue.get()
                self.record_queue_lengths()

                service_start = self.env.now
                wait_time = service_start - customer["arrival_time"]

                self.wait_times_all.append(wait_time)
                self.wait_times_walkin.append(wait_time)

                service_time = self.sample_kiosk_service()
                self.total_kiosk_busy_time += service_time

                yield self.env.timeout(service_time)

                departure_time = self.env.now
                system_time = departure_time - customer["arrival_time"]

                self.system_times_all.append(system_time)
                self.system_times_walkin.append(system_time)

                self.completed_walkins += 1
                self.completed_customers += 1
            else:
                yield self.env.timeout(0.1)

    def time_weighted_avg_queue_lengths(self):
        appt_area = 0.0
        walkin_area = 0.0

        for i in range(len(self.queue_length_timeline) - 1):
            t0, a0, w0 = self.queue_length_timeline[i]
            t1, _, _ = self.queue_length_timeline[i + 1]
            dt = t1 - t0
            appt_area += a0 * dt
            walkin_area += w0 * dt

        last_time, last_a, last_w = self.queue_length_timeline[-1]
        if last_time < self.sim_time:
            dt = self.sim_time - last_time
            appt_area += last_a * dt
            walkin_area += last_w * dt

        return {
            "avg_appt_queue": appt_area / self.sim_time,
            "avg_walkin_queue": walkin_area / self.sim_time,
            "avg_total_queue": (appt_area + walkin_area) / self.sim_time,
            "max_appt_queue": max(a for _, a, _ in self.queue_length_timeline),
            "max_walkin_queue": max(w for _, _, w in self.queue_length_timeline),
            "max_total_queue": max(a + w for _, a, w in self.queue_length_timeline),
        }

    def safe_mean(self, values):
        return statistics.mean(values) if values else 0.0

    def results(self):
        qstats = self.time_weighted_avg_queue_lengths()

        clerk_capacity_time = self.num_clerks * self.sim_time
        kiosk_capacity_time = self.num_kiosks * self.sim_time
        total_capacity_time = (self.num_clerks + self.num_kiosks) * self.sim_time

        clerk_util = self.total_clerk_busy_time / clerk_capacity_time if clerk_capacity_time else 0.0
        kiosk_util = self.total_kiosk_busy_time / kiosk_capacity_time if kiosk_capacity_time else 0.0
        combined_util = (
            (self.total_clerk_busy_time + self.total_kiosk_busy_time) / total_capacity_time
            if total_capacity_time else 0.0
        )

        return {
            "completed_customers": self.completed_customers,
            "completed_appointments": self.completed_appointments,
            "completed_walkins": self.completed_walkins,

            "avg_wait_all": self.safe_mean(self.wait_times_all),
            "avg_wait_appt": self.safe_mean(self.wait_times_appt),
            "avg_wait_walkin": self.safe_mean(self.wait_times_walkin),

            "avg_system_all": self.safe_mean(self.system_times_all),
            "avg_system_appt": self.safe_mean(self.system_times_appt),
            "avg_system_walkin": self.safe_mean(self.system_times_walkin),

            "clerk_utilization": clerk_util,
            "kiosk_utilization": kiosk_util,
            "combined_utilization": combined_util,

            **qstats
        }


def run_simulation():
    SIM_TIME = 480

    env = simpy.Environment()
    sim = DMVPrioritySimulation(
        env,
        num_clerks=3,
        num_kiosks=2,
        sim_time=SIM_TIME,
        appt_interarrival=(2, 4),      # appointments arrive less often
        walkin_interarrival=(0.8, 1.5),# walk-ins heavier traffic
        clerk_service_time=(8, 15),
        kiosk_service_time=(4, 8),
        seed=42
    )

    # Start arrival processes
    env.process(sim.appointment_arrivals())
    env.process(sim.walkin_arrivals())

    # Start clerk processes
    for i in range(sim.num_clerks):
        env.process(sim.clerk_process(i))

    # Start kiosk processes
    for i in range(sim.num_kiosks):
        env.process(sim.kiosk_process(i))

    env.run(until=SIM_TIME)

    results = sim.results()

    print("\nDMV Priority Simulation Results")
    print("-" * 40)
    print(f"Completed customers:          {results['completed_customers']}")
    print(f"Completed appointments:       {results['completed_appointments']}")
    print(f"Completed walk-ins:           {results['completed_walkins']}")
    print()
    print(f"Average appointment queue:    {results['avg_appt_queue']:.2f}")
    print(f"Average walk-in queue:        {results['avg_walkin_queue']:.2f}")
    print(f"Average total queue:          {results['avg_total_queue']:.2f}")
    print(f"Max appointment queue:        {results['max_appt_queue']}")
    print(f"Max walk-in queue:            {results['max_walkin_queue']}")
    print(f"Max total queue:              {results['max_total_queue']}")
    print()
    print(f"Average wait (all):           {results['avg_wait_all']:.2f} minutes")
    print(f"Average wait (appointments):  {results['avg_wait_appt']:.2f} minutes")
    print(f"Average wait (walk-ins):      {results['avg_wait_walkin']:.2f} minutes")
    print()
    print(f"Average system time (all):    {results['avg_system_all']:.2f} minutes")
    print(f"Average system time (appt):   {results['avg_system_appt']:.2f} minutes")
    print(f"Average system time (walkin): {results['avg_system_walkin']:.2f} minutes")
    print()
    print(f"Clerk utilization:            {results['clerk_utilization']:.2%}")
    print(f"Kiosk utilization:            {results['kiosk_utilization']:.2%}")
    print(f"Combined utilization:         {results['combined_utilization']:.2%}")


if __name__ == "__main__":
    run_simulation()
