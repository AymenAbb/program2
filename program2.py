# Note: This was 99% done by o4 mini high, I have only double checked snippets of code and re-asserted to the AI to follow the document provided. 
# Initial prompt was the original document in full.

import random
import numpy as np

TIMES = ['10 AM', '11 AM', '12 PM', '1 PM', '2 PM', '3 PM']
TIME_MAP = {'10 AM': 10, '11 AM': 11, '12 PM': 12, '1 PM': 13, '2 PM': 14, '3 PM': 15}
ROOMS = [
    {'name': 'Slater 003', 'capacity': 45},
    {'name': 'Roman 216', 'capacity': 30},
    {'name': 'Loft 206', 'capacity': 75},
    {'name': 'Roman 201', 'capacity': 50},
    {'name': 'Loft 310', 'capacity': 108},
    {'name': 'Beach 201', 'capacity': 60},
    {'name': 'Beach 301', 'capacity': 75},
    {'name': 'Logos 325', 'capacity': 450},
    {'name': 'Frank 119', 'capacity': 60},
]
FACILITATORS = ['Lock', 'Glen', 'Banks', 'Richards', 'Shaw', 'Singer', 'Uther', 'Tyler', 'Numen', 'Zeldin']
ACTIVITIES = [
    {'name': 'SLA100A', 'expected': 50, 'preferred': ['Glen', 'Lock', 'Banks', 'Zeldin'], 'others': ['Numen', 'Richards']},
    {'name': 'SLA100B', 'expected': 50, 'preferred': ['Glen', 'Lock', 'Banks', 'Zeldin'], 'others': ['Numen', 'Richards']},
    {'name': 'SLA191A', 'expected': 50, 'preferred': ['Glen', 'Lock', 'Banks', 'Zeldin'], 'others': ['Numen', 'Richards']},
    {'name': 'SLA191B', 'expected': 50, 'preferred': ['Glen', 'Lock', 'Banks', 'Zeldin'], 'others': ['Numen', 'Richards']},
    {'name': 'SLA201',  'expected': 50, 'preferred': ['Glen', 'Banks', 'Zeldin', 'Shaw'], 'others': ['Numen', 'Richards', 'Singer']},
    {'name': 'SLA291',  'expected': 50, 'preferred': ['Lock', 'Banks', 'Zeldin', 'Singer'], 'others': ['Numen', 'Richards', 'Shaw', 'Tyler']},
    {'name': 'SLA303',  'expected': 60, 'preferred': ['Glen', 'Zeldin', 'Banks'], 'others': ['Numen', 'Singer', 'Shaw']},
    {'name': 'SLA304',  'expected': 25, 'preferred': ['Glen', 'Banks', 'Tyler'], 'others': ['Numen', 'Singer', 'Shaw', 'Richards', 'Uther', 'Zeldin']},
    {'name': 'SLA394',  'expected': 20, 'preferred': ['Tyler', 'Singer'], 'others': ['Richards', 'Zeldin']},
    {'name': 'SLA449',  'expected': 60, 'preferred': ['Tyler', 'Singer', 'Shaw'], 'others': ['Zeldin', 'Uther']},
    {'name': 'SLA451',  'expected': 100,'preferred': ['Tyler', 'Singer', 'Shaw'], 'others': ['Zeldin', 'Uther', 'Richards', 'Banks']},
]

class Schedule:
    def __init__(self, assignments=None):
        if assignments:
            self.assignments = assignments.copy()
        else:
            self.assignments = {}
            for act in ACTIVITIES:
                self.assignments[act['name']] = {
                    'time':  random.choice(TIMES),
                    'room':  random.choice(ROOMS)['name'],
                    'fac':   random.choice(FACILITATORS)
                }

    def compute_fitness(self):
        fitness = 0.0
        fac_times = {}
        fac_counts = {}
        # Tally facilitator times and counts
        for act, info in self.assignments.items():
            fac = info['fac']
            fac_times.setdefault(fac, []).append(info['time'])
            fac_counts[fac] = fac_counts.get(fac, 0) + 1

        # Evaluate each activity
        for act, info in self.assignments.items():
            time, room, fac = info['time'], info['room'], info['fac']
            act_def = next(a for a in ACTIVITIES if a['name'] == act)

            # Room/time collision
            collisions = sum(
                1 for o, oi in self.assignments.items()
                if o != act and oi['time'] == time and oi['room'] == room
            )
            if collisions > 0:
                fitness -= 0.5

            # Room size scoring
            cap = next(r['capacity'] for r in ROOMS if r['name'] == room)
            exp = act_def['expected']
            if cap < exp:
                fitness -= 0.5
            elif cap > 6 * exp:
                fitness -= 0.4
            elif cap > 3 * exp:
                fitness -= 0.2
            else:
                fitness += 0.3

            # Facilitator match
            if fac in act_def['preferred']:
                fitness += 0.5
            elif fac in act_def['others']:
                fitness += 0.2
            else:
                fitness -= 0.1

            # Facilitator time-slot load
            same_slot = fac_times[fac].count(time)
            if same_slot == 1:
                fitness += 0.2
            else:
                fitness -= 0.2

            # Facilitator total load
            total = fac_counts[fac]
            if total > 4:
                fitness -= 0.5
            elif total in (1, 2) and not (fac == 'Tyler' and total < 2):
                fitness -= 0.4

        # Activity-specific time adjustments
        def time_diff(a, b):
            return abs(TIME_MAP[self.assignments[a]['time']] - TIME_MAP[self.assignments[b]['time']])

        # SLA100 sections
        d1 = time_diff('SLA100A', 'SLA100B')
        if d1 >= 4:
            fitness += 0.5
        if d1 == 0:
            fitness -= 0.5

        # SLA191 sections
        d2 = time_diff('SLA191A', 'SLA191B')
        if d2 >= 4:
            fitness += 0.5
        if d2 == 0:
            fitness -= 0.5

        # Cross-activity adjustments between SLA191 and SLA100
        for a in ['SLA191A', 'SLA191B']:
            for b in ['SLA100A', 'SLA100B']:
                dt = time_diff(a, b)
                if dt == 1:
                    fitness += 0.5
                    # Building separation penalty
                    room_a = self.assignments[a]['room']
                    room_b = self.assignments[b]['room']
                    bld = lambda x: x.startswith('Roman') or x.startswith('Beach')
                    if bld(room_a) != bld(room_b):
                        fitness -= 0.4
                elif dt == 2:
                    fitness += 0.25
                elif dt == 0:
                    fitness -= 0.25

        return fitness

    def mutate(self, rate):
        for act in self.assignments:
            if random.random() < rate:
                field = random.choice(['time', 'room', 'fac'])
                if field == 'time':
                    self.assignments[act]['time'] = random.choice(TIMES)
                elif field == 'room':
                    self.assignments[act]['room'] = random.choice(ROOMS)['name']
                else:
                    self.assignments[act]['fac'] = random.choice(FACILITATORS)

    @staticmethod
    def crossover(p1, p2):
        assigns = {}
        for act in p1.assignments:
            src = p1 if random.random() < 0.5 else p2
            assigns[act] = src.assignments[act].copy()
        return Schedule(assignments=assigns)

    def save(self, filename):
        with open(filename, 'w') as f:
            for act, info in sorted(self.assignments.items()):
                f.write(f"{act}: {info['time']}, {info['room']}, {info['fac']}\n")


class GeneticAlgorithm:
    def __init__(self, pop_size=500, mutation_rate=0.01):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

    def softmax(self, vals):
        ex = np.exp(vals - np.max(vals))
        return ex / ex.sum()

    def run(self):
        pop = [Schedule() for _ in range(self.pop_size)]
        fitnesses = np.array([s.compute_fitness() for s in pop])
        avg100 = None
        gen = 1

        # Evolve until improvement over gen 100 is below 1%
        while True:
            avg = fitnesses.mean()
            if gen == 100:
                avg100 = avg
            if gen > 100 and (avg - avg100) < 0.01 * avg100:
                break

            probs = self.softmax(fitnesses)
            new_pop = []
            while len(new_pop) < self.pop_size:
                idx = np.random.choice(self.pop_size, 2, replace=False, p=probs)
                p1, p2 = pop[idx[0]], pop[idx[1]]
                c1 = Schedule.crossover(p1, p2)
                c2 = Schedule.crossover(p1, p2)
                c1.mutate(self.mutation_rate)
                c2.mutate(self.mutation_rate)
                new_pop.extend([c1, c2])

            pop = new_pop[:self.pop_size]
            fitnesses = np.array([s.compute_fitness() for s in pop])
            gen += 1

        # Report and output best
        best_idx = int(np.argmax(fitnesses))
        best = pop[best_idx]
        best.save('best_schedule.txt')
        print('Best fitness:', fitnesses[best_idx])
        print('\nBest schedule:')
        for act, info in sorted(best.assignments.items()):
            print(f"{act}: {info['time']}, {info['room']}, {info['fac']}")


if __name__ == '__main__':
    ga = GeneticAlgorithm()
    ga.run()
