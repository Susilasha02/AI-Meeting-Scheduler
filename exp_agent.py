# exp_agent.py
def explain(slot):
    b=slot["meta"]["buffers"]
    return f"Conflict‑free; honors {b['pre']}m buffers; within work hours."
