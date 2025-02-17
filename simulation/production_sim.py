"""
Module for production simulation using SimPy.
"""
import simpy

class OperationBasedProductionSimulation:
    """Simulates production operations with resources and constraints."""
    def __init__(self, env, jobs):
        self.env = env
        self.jobs = jobs
        machines = set()
        for job in jobs:
            for op in job["Operationen"]:
                machines.add(op["Maschine"])
        self.machines = {m: simpy.Resource(env, capacity=1) for m in machines}
        self.waiting_jobs = list(jobs)
        self.finished_jobs = 0
        self.current_makespan = 0
        self.job_finish_times = {}
        self.total_cost = 0

        # Production aids setup
        production_aid_types = ["Schablone", "Öl", "Kühlmittel", "Werkzeug"]
        self.production_aids = {aid: simpy.Container(env, init=5, capacity=5) for aid in production_aid_types}
        self.MAX_AID = 5
        self.REORDER_TIME = 10
        self.ORDER_COST = 20
        self.reorder_log = []

    def process_job(self, job):
        """Process a single job through all its operations."""
        job_cost = 0
        for op in job["Operationen"]:
            if "benötigteHilfsmittel" in op:
                for aid in op["benötigteHilfsmittel"]:
                    if self.production_aids[aid].level < 1:
                        event = {
                            "time": self.env.now,
                            "aid": aid,
                            "action": "reorder",
                            "order_cost": self.ORDER_COST
                        }
                        self.reorder_log.append(event)
                        yield self.env.timeout(self.REORDER_TIME)
                        job_cost += self.ORDER_COST
                        missing = self.MAX_AID - self.production_aids[aid].level
                        if missing > 0:
                            yield self.production_aids[aid].put(missing)
                    yield self.production_aids[aid].get(1)

            processing_time = op["benötigteZeit"]
            if "umruestzeit" in op:
                processing_time += op["umruestzeit"]
                job_cost += op.get("umruestkosten", 0)
            if "zwischenlager" in op:
                wait_time = op["zwischenlager"]["minVerweildauer"]
                processing_time += wait_time
                job_cost += op["zwischenlager"]["lagerkosten"] * wait_time
            machine = op["Maschine"]
            with self.machines[machine].request() as req:
                yield req
                yield self.env.timeout(processing_time)

        finish_time = self.env.now
        if finish_time > self.current_makespan:
            self.current_makespan = finish_time
        self.finished_jobs += 1
        self.job_finish_times[job["Name"]] = finish_time
        self.total_cost += job_cost

    def schedule_job(self, job):
        """Schedule a job for processing."""
        yield self.env.process(self.process_job(job))
