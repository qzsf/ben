# more randomness introduced
# - random no show,
#   service time,
#   walk in arrival rate fluctuation,
#   appointment arrival rate fluctuation,

import simpy
import random
import math
import json

import matplotlib.pyplot as plt

# Configuration
NUM_CLERKS = 10
NUM_KIOSKS = 2
USE_KIOSKS = True

SLOT_DURATION = 60      # Minutes per appointment block
APPS_PER_SLOT = 5       # 5 clerks reserved for appointments
GRACE_PERIOD = 5        # 5-minute wait before walk-ins can take reserved spots
AVG_SERVICE_TIME = 60   # Average time a clerk spends with a customer
SIM_TIME = 420          # 7-hour workday (in minutes)

PRE_OPENING_LINE = 10           # walk-in pre-opening line of 10 people
APPT_NO_SHOW_RATE = 0.1         # 10% appointment no-show
# WALKIN_PATIENCE = 60            # the time walk-ins become impatient

ID = 0                  # visitor id
WALKIN_TOTAL = 0        # track total number of walk-ins. some may leave without being served
WALKIN_RATE = 10        # number of walk-ins per hour

class FO:
    def __init__(self, env):
        self.env = env
        # PriorityResource: 0 is highest priority (Appointments)
        self.clerks = simpy.PriorityResource(env, capacity=NUM_CLERKS)
        self.kiosks = simpy.Resource(env, capacity=NUM_KIOSKS)
        self.served_apps = 0
        self.served_walkins = 0
        self.queue_length = 0

    def service_process(self, name, resource_type):
        """Simulates the actual time spent at the window or kiosk."""
        if resource_type == 'kiosk':
            # duration = max(5, random.normalvariate(AVG_SERVICE_TIME * 0.5, 1))  # Kiosks are faster
            duration = max(5, random.normalvariate(AVG_SERVICE_TIME, 1))
        else:
            duration = max(5, random.normalvariate(AVG_SERVICE_TIME, 1))
        yield self.env.timeout(duration)

# calculate the arrival rate at a moment
# use Gaussian bell-shape to mimic a day traffic
def walkin_rate(minute):
    # pre-opening line of 10-person
    # if minute == 0:
    #     return max(0, int(round(random.gauss(PRE_OPENING_LINE, 3))))

    # Parameters for the bell curve
    peak_time = SIM_TIME / 2  # Peak at midday
    peak_rate = 0.15           # Peak arrivals per minute (adjust as needed)
    stddev = SIM_TIME / 5     # Spread of the bell curve

    # Gaussian curves
    # morning bell curve
    morning_peak_time = 0   # at opening
    morning_peak_rate = 0.15
    morning_stddev = 25     # 30 minutes wide
    morning = morning_peak_rate * math.exp(-((minute - morning_peak_time) ** 2) / (2 * morning_stddev ** 2))

    # main bell shape
    midday = peak_rate * math.exp(-((minute - peak_time) ** 2) / (2 * stddev ** 2))

    base = morning + midday

    # Add a small random fluctuation (e.g., ±10% of the base rate)
    fluctuation = random.uniform(-0.1, 0.1) * base
    return max(0, base + fluctuation)  # Ensure rate is not negative

# constant walk in rate
def walkin_rate_flat(m):
    return 60/WALKIN_RATE     # 1 walk-in every 6 minutes

# generate rate data for plotting
minutes = list(range(SIM_TIME))
rates = [walkin_rate(m) for m in minutes]
# plot
plt.figure(figsize=(10, 4))
plt.plot(minutes, rates, label="Walk-in Arrival Rate")
plt.xlabel("Minutes since opening")
plt.ylabel("Arrival rate (per minute)")
plt.title("Walk-in Arrival Rate")
plt.grid(True)
plt.legend()
plt.show()

# bar chart
BIN_SIZE = 10    # Bin size in minutes
# Aggregate into 10-minute bins
bins = []
for i in range(0, SIM_TIME, BIN_SIZE):
    bin_total = sum(rates[i:i+BIN_SIZE])
    bins.append(bin_total)

# add pre-opening line to the first bin
# bins[0] += 10   # number of persons in line

# Plot bar chart
# bin_labels = [f"{i}-{i+BIN_SIZE}" for i in range(0, SIM_TIME, BIN_SIZE)]
# plt.figure(figsize=(10, 4))
# plt.bar(bin_labels, bins, color='skyblue')
# plt.xlabel("Minutes since opening (10-min bins)")
# plt.ylabel("Number of walk-in visitors")
# plt.title("Walk-in Visitors Every 10 Minutes (flat)")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.legend()
# plt.show()

def appointment_customer(env, fo, slot_start, id):
    """Logic for customers with an appointment."""
    # Simulate arrival: Some are early, some are slightly late
    arrival_time = slot_start + random.normalvariate(0, 4)
    # print(arrival_time)

    # if a customer comes on or before appointment time
    #    gets in the queue at the slot_start time
    # else
    #    enters the queue when arrives
    wait_to_arrive = max(0, arrival_time - env.now)
    yield env.timeout(wait_to_arrive)

    # if arrive within 5 mins of slot start to keep priority
    if env.now <= slot_start + GRACE_PERIOD:
        priority = 0 # High priority
    else:
        priority = 1 # Becomes like a walk-in if late

    clerks_count = fo.clerks.count
    with fo.clerks.request(priority=priority) as req:
        queue_enter_time = env.now
        queue_length = fo.queue_length
        yield req
        fo.queue_length -= 1
        service_start = env.now
        wait_time = service_start - queue_enter_time
        yield env.process(fo.service_process("Appointment", "clerk"))
        departure_time = env.now
        fo.served_apps += 1

    global ID
    ID += 1

    # record stats
    record = {
        "id": ID,
        "name": f"a-{id}",
        "type": 'appointment',
        "resource": "clerk",
        "slot": slot_start,
        "queue_length": queue_length,
        "clerk_count": clerks_count,
        "arrival_time": queue_enter_time,
        "service_start": service_start,
        "wait_time": wait_time,
        "service_time": departure_time - service_start,
        "time_in_system": departure_time - queue_enter_time,
        "departure": departure_time
    }

    stats["visitors"].append(record)

def cancel_or_release(resource, request):
    if request.triggered:
        resource.release(request)
    else:
        request.cancel()

def walkin_customer(env, fo, id):
    global ID
    # print("Walkin ID:", id)
    # print(len(fo.clerks.queue))

    # simulate reneging
    queue_length = fo.queue_length
    clerks_count = fo.clerks.count

    # Create the request objects
    req_kiosk = fo.kiosks.request()
    req_clerk = fo.clerks.request(priority=2)

    queue_enter_time = env.now
    patience_timer = env.timeout(100)

    requests = [req_clerk, patience_timer]
    if USE_KIOSKS:
        requests.append(req_kiosk)

    # result = yield env.any_of([req_clerk, req_kiosk, patience_timer])
    result = yield env.any_of(requests)

    service_start = env.now
    wait_time = service_start - queue_enter_time

    if req_clerk in result and req_kiosk in result:
        fo.queue_length -= 1
        service_start = env.now
        resource_type = ''
        if random.choice(["clerk", "kiosk"]) == "clerk":
            fo.kiosks.release(req_kiosk)
            resource_type = 'clerk'
            # print(f"{id} (Priority 2) got a CLERK at {env.now}")
            yield env.process(fo.service_process("Walk-in", "clerk"))
            departure_time = env.now
            fo.served_walkins += 1
            fo.clerks.release(req_clerk)
        else:
            fo.clerks.release(req_clerk)
            resource_type = 'kiosk'
            # print(f"{id} got a KIOSK at {env.now}")
            yield env.process(fo.service_process("Walk-in", "kiosk"))
            departure_time = env.now
            fo.served_walkins += 1
            fo.kiosks.release(req_kiosk)

        ID += 1

        # record stats
        record = {
            "id": ID,
            "name": f"w-{id}",
            "type": 'walkin',
            "resource": resource_type,
            "slot": '',
            "arrival_time": queue_enter_time,
            "service_start": service_start,
            "queue_length": queue_length,
            "clerk_count": clerks_count,
            "wait_time": wait_time,
            "service_time": departure_time - service_start,
            "time_in_system": departure_time - queue_enter_time,
            "departure": departure_time
        }
        stats["visitors"].append(record)

    elif req_clerk in result:
        fo.queue_length -= 1
        service_start = env.now
        cancel_or_release(fo.kiosks, req_kiosk)
        # print(f"{id} (Priority 2) got a CLERK at {env.now}")
        yield env.process(fo.service_process("Walk-in", "clerk"))
        departure_time = env.now
        fo.served_walkins += 1
        fo.clerks.release(req_clerk)

        ID += 1

        # record stats
        record = {
            "id": ID,
            "name": f"w-{id}",
            "type": 'walkin',
            "resource": "clerk",
            "slot": '',
            "arrival_time": queue_enter_time,
            "service_start": service_start,
            "queue_length": queue_length,
            "clerk_count": clerks_count,
            "wait_time": wait_time,
            "service_time": departure_time - service_start,
            "time_in_system": departure_time - queue_enter_time,
            "departure": departure_time
        }
        stats["visitors"].append(record)


    elif req_kiosk in result:
        fo.queue_length -= 1
        service_start = env.now
        cancel_or_release(fo.clerks, req_clerk)
        # print(f"{id} got a KIOSK at {env.now}")
        yield env.process(fo.service_process("Walk-in", "kiosk"))
        departure_time = env.now
        fo.served_walkins += 1
        fo.kiosks.release(req_kiosk)

        ID += 1

        # record stats
        record = {
            "id": ID,
            "name": f"w-{id}",
            "type": 'walkin',
            "resource": "kiosk",
            "slot": '',
            "arrival_time": queue_enter_time,
            "service_start": service_start,
            "queue_length": queue_length,
            "clerk_count": clerks_count,
            "wait_time": wait_time,
            "service_time": departure_time - service_start,
            "time_in_system": departure_time - queue_enter_time,
            "departure": departure_time
        }
        stats["visitors"].append(record)

    else:
        # Timeout triggered
        req_clerk.cancel()
        req_kiosk.cancel()
        fo.queue_length -= 1
        # print(f"{id} left the line (Time limit reached)")



    # with fo.clerks.request(priority=2) as req:
    #     queue_enter_time = env.now
    #     yield req
    #     service_start = env.now
    #     wait_time = service_start - queue_enter_time
    #     yield env.process(fo.service_process("Walk-in"))
    #     departure_time = env.now
    #     fo.served_walkins += 1

    # global ID
    # ID += 1

    # # record stats
    # record = {
    #     "id": ID,
    #     "name": f"w-{id}",
    #     "type": 'walkin',
    #     "slot": '',
    #     "arrival_time": queue_enter_time,
    #     "service_start": service_start,
    #     "queue_length": len(fo.clerks.queue),
    #     "clerk_count": clerks_count,
    #     "wait_time": wait_time,
    #     "service_time": departure_time - service_start,
    #     "time_in_system": departure_time - queue_enter_time,
    #     "departure": departure_time
    # }
    # stats["visitors"].append(record)
        # else:
            # print(f"Walk-in w-{id} left after waiting {env.now - queue_enter_time:.0f} minutes")

def appointment_generator(env, fo):
    id = 0
    """Triggers appointment slots every 30 minutes."""
    while env.now < SIM_TIME:
        slot_start = env.now
        for _ in range(APPS_PER_SLOT):
            # simulate no-show
            if random.random() < APPT_NO_SHOW_RATE:
                continue  # Skip this appointment (no-show)
            id += 1
            fo.queue_length += 1
            env.process(appointment_customer(env, fo, slot_start, id))
        yield env.timeout(SLOT_DURATION)

def walkin_generator(env, fo):
    global WALKIN_TOTAL
    id = 0
    # wait appt visitors for 5 minutes
    # yield env.timeout(3)

    # Pre-opening line: generate 10 walk-ins at time 0
    # for _ in range(10):
    # for _ in range(random.randint(PRE_OPENING_LINE-2, PRE_OPENING_LINE+2)):
    #     WALKIN_TOTAL += 1
    #     id += 1
    #     env.process(walkin_customer(env, fo, id))
    #     yield env.timeout(random.normalvariate(0.5, 0.1))

    """Simulates random walk-in arrivals."""
    while env.now < SIM_TIME:
        WALKIN_TOTAL += 1
        # rate = walkin_rate_flat(env.now)
        rate = 1 / walkin_rate(env.now)
        # Average one walk-in every 10 minutes
        # yield env.timeout(random.expovariate(1/10))
        # yield env.timeout(random.expovariate(rate))
        id += 1
        fo.queue_length += 1
        env.process(walkin_customer(env, fo, id))
        yield env.timeout(rate)

# Run the simulation
env = simpy.Environment()
field_office = FO(env)

stats = {
        "visitors": []
    }

env.process(appointment_generator(env, field_office))
env.process(walkin_generator(env, field_office))
# extra 30 minutes to finish processing all cases
env.run(until = SIM_TIME + 30)

print(f"--- End of Day Report ---")
print(f"Appointments served: {field_office.served_apps}")
print(f"Walk-ins total: {WALKIN_TOTAL}")
print(f"Walk-ins served: {field_office.served_walkins}")

# print(stats['visitors'])
with open('visitors.json', 'w') as f:
    json.dump(stats, f, indent=4)

appt_num = 0
appt_wait_sum = 0
walkin_num = 0
walkin_wait_sum = 0
for visitor in stats["visitors"]:
    if visitor["type"] == "appointment":
        appt_num += 1
        appt_wait_sum += visitor["wait_time"]
    else:
        walkin_num += 1
        walkin_wait_sum += visitor["wait_time"]

# print(appt_num)
# print(walkin_num)
print("Appt average wait time:", appt_wait_sum/appt_num)
print("Walkin average wait time:", walkin_wait_sum/walkin_num)


