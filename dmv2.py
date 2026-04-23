import simpy
import random
import statistics


class DMVOneLineSimulation:
    def __init__(
        self,
        env,
        num_clerks=3,
        num_kiosks=2,
        sim_time=480,
        appt_interarrival=(2, 4),
        walkin_interarrival=(0.5, 1.5),
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

        # 🔥 ONE shared line
        self.queue = simpy.FilterStore(env)

        # Metrics
        self.completed = 0
        self.wait_times = []
        self.system_times = []

        self.wait_appt = []
        self.wait_walkin = []

        self.total_clerk_busy = 0.0
        self.total_kiosk_busy = 0.0

        self.queue_timeline = [(0, 0)]
        self.customer_id = 0

    def next_id(self):
        self.customer_id += 1
        return self.customer_id

    def record_queue(self):
        self.queue_timeline.append((self.env.now, len(self.queue.items)))

    def sample_appt_arrival(self):
        return random.uniform(*self.appt_interarrival)

    def sample_walkin_arrival(self):
        return random.uniform(*self.walkin_interarrival)

    def sample_clerk_service(self):
        return random.uniform(*self.clerk_service_time)

    def sample_kiosk_service(self):
        return random.uniform(*self.kiosk_service_time)

    # -------- arrivals --------

    def appointment_arrivals(self):
        while True:
            yield self.env.timeout(self.sample_appt_arrival())

            customer = {
                "id": self.next_id(),
                "type": "appointment",
                "arrival": self.env.now
            }

            yield self.queue.put(customer)
            self.record_queue()

    def walkin_arrivals(self):
        while True:
            yield self.env.timeout(self.sample_walkin_arrival())

            customer = {
                "id": self.next_id(),
                "type": "walkin",
                "arrival": self.env.now
            }

            yield self.queue.put(customer)
            self.record_queue()

    # -------- servers --------

    def clerk(self, cid):
        while True:
            # 🔥 clerks prefer appointments first
            if any(c["type"] == "appointment" for c in self.queue.items):
                customer = yield self.queue.get(lambda c: c["type"] == "appointment")
            else:
                customer = yield self.queue.get(lambda c: True)

            self.record_queue()

            start = self.env.now
            wait = start - customer["arrival"]

            self.wait_times.append(wait)
            if customer["type"] == "appointment":
                self.wait_appt.append(wait)
            else:
                self.wait_walkin.append(wait)

            service = self.sample_clerk_service()
            self.total_clerk_busy += service

            yield self.env.timeout(service)

            system = self.env.now - customer["arrival"]
            self.system_times.append(system)
            self.completed += 1

    def kiosk(self, kid):
        while True:
            # 🔥 kiosks ONLY take walk-ins
            customer = yield self.queue.get(lambda c: c["type"] == "walkin")

            self.record_queue()

            start = self.env.now
            wait = start - customer["arrival"]

            self.wait_times.append(wait)
            self.wait_walkin.append(wait)

            service = self.sample_kiosk_service()
            self.total_kiosk_busy += service

            yield self.env.timeout(service)

            system = self.env.now - customer["arrival"]
            self.system_times.append(system)
            self.completed += 1

    # -------- metrics --------

    def avg_queue(self):
        area = 0
        for i in range(len(self.queue_timeline) - 1):
            t0, q0 = self.queue_timeline[i]
            t1, _ = self.queue_timeline[i + 1]
            area += q0 * (t1 - t0)

        return area / self.sim_time

    def results(self):
        clerk_util = self.total_clerk_busy / (self.num_clerks * self.sim_time)
        kiosk_util = self.total_kiosk_busy / (self.num_kiosks * self.sim_time)

        return {
            "completed": self.completed,
            "avg_wait": statistics.mean(self.wait_times),
            "avg_wait_appt": statistics.mean(self.wait_appt) if self.wait_appt else 0,
            "avg_wait_walkin": statistics.mean(self.wait_walkin) if self.wait_walkin else 0,
            "avg_system": statistics.mean(self.system_times),
            "avg_queue": self.avg_queue(),
            "clerk_util": clerk_util,
            "kiosk_util": kiosk_util
        }


def run():
    env = simpy.Environment()

    sim = DMVOneLineSimulation(env)

    env.process(sim.appointment_arrivals())
    env.process(sim.walkin_arrivals())

    for i in range(sim.num_clerks):
        env.process(sim.clerk(i))

    for i in range(sim.num_kiosks):
        env.process(sim.kiosk(i))

    env.run(until=sim.sim_time)

    r = sim.results()

    print("\nONE-LINE DMV RESULTS")
    print("-" * 35)
    print(f"Completed:           {r['completed']}")
    print(f"Avg wait (all):      {r['avg_wait']:.2f}")
    print(f"Avg wait (appt):     {r['avg_wait_appt']:.2f}")
    print(f"Avg wait (walkin):   {r['avg_wait_walkin']:.2f}")
    print(f"Avg system time:     {r['avg_system']:.2f}")
    print(f"Avg queue length:    {r['avg_queue']:.2f}")
    print(f"Clerk utilization:   {r['clerk_util']:.2%}")
    print(f"Kiosk utilization:   {r['kiosk_util']:.2%}")


if __name__ == "__main__":
    run()
