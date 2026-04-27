import random

import simpy


def cancel_or_release(resource, request):
    if request.triggered:
        resource.release(request)
    else:
        request.cancel()


def visitor(env, name, priority, clerks, kiosks):
    # 1. Create priority requests
    # Clerks now care about the priority value
    req_clerk = clerks.request(priority=priority)
    req_kiosk = kiosks.request()  # Kiosks usually first-come-first-served

    # 2. The race: Wait for a clerk, a kiosk, or the 30-min exit
    patience_timer = env.timeout(30)
    result = yield env.any_of([req_clerk, req_kiosk, patience_timer])

    if req_clerk in result and req_kiosk in result:
        if random.choice(["clerk", "kiosk"]) == "clerk":
            kiosks.release(req_kiosk)
            print(f"{name} (Priority {priority}) got a CLERK at {env.now}")
            yield env.timeout(15)
            clerks.release(req_clerk)
        else:
            clerks.release(req_clerk)
            print(f"{name} got a KIOSK at {env.now}")
            yield env.timeout(15)
            kiosks.release(req_kiosk)

    elif req_clerk in result:
        cancel_or_release(kiosks, req_kiosk)
        print(f"{name} (Priority {priority}) got a CLERK at {env.now}")
        yield env.timeout(15)
        clerks.release(req_clerk)

    elif req_kiosk in result:
        cancel_or_release(clerks, req_clerk)
        print(f"{name} got a KIOSK at {env.now}")
        yield env.timeout(15)
        kiosks.release(req_kiosk)

    else:
        # Timeout triggered
        req_clerk.cancel()
        req_kiosk.cancel()
        print(f"{name} left the line (Time limit reached)")


# Setup
env = simpy.Environment()
clerks = simpy.PriorityResource(env, capacity=4)
kiosks = simpy.Resource(env, capacity=2)

# Example: High priority visitor (0) vs Low priority visitor (1)
env.process(visitor(env, "VIP Visitor", 0, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))
env.process(visitor(env, "Standard Visitor", 1, clerks, kiosks))

env.run()
