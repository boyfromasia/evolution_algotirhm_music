from mido import MidiFile, Message, MidiTrack, merge_tracks
import random
from copy import deepcopy

path = "barbiegirl_mono.mid"

num_to_str = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
str_to_num = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
major = [2, 2, 1, 2, 2, 2, 1]
minor = [2, 1, 2, 2, 1, 2, 2]

POPULATION_SIZE = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATION = 1000


# Get notes from midi file.
def get_notes(mid: MidiFile, octave=True, unique=False, num=False, by_strong_beat=False) -> list:
    notes = []
    tick = mid.ticks_per_beat

    strong_beat = []
    cur_time = 0
    cur_strong_beat_num = 0

    for el in mid.tracks[1]:
        if isinstance(el, Message):
            if by_strong_beat is False:
                if el.type == "note_on":
                    if num is False:
                        note = num_to_str[el.note % 12]
                    else:
                        note = el.note % 12

                    if octave:
                        note += str(el.note // 12)

                    if unique and note not in notes:
                        notes.append(note)
                    elif unique is False:
                        notes.append(note)

            else:
                cur_time += el.time
                if el.type == "note_off":
                    if num is False:
                        note = num_to_str[el.note % 12]
                    else:
                        note = el.note % 12

                    if octave:
                        note += str(el.note // 12)

                    if unique and note not in notes:
                        strong_beat.append(note)
                    elif unique is False:
                        strong_beat.append(note)

                if cur_strong_beat_num != cur_time // (tick * 2):
                    for k in range((cur_time // (tick * 2)) - cur_strong_beat_num):
                        notes.append(strong_beat)
                        strong_beat = []

                cur_strong_beat_num = cur_time // (tick * 2)

        if el == mid.tracks[1][-1] and strong_beat != []:
            notes.append(strong_beat)

    return notes


# Get octave of first note. Use for write chord to midi file if strong beat have no notes.
def get_octave_of_first_note(mid: MidiFile) -> int:
    for x in mid.tracks[1]:
        if isinstance(x, Message):
            if x.type == "note_on":
                return x.note // 12 - 2 if x.note // 12 - 2 > 0 else 0


# Get gamma for defining key of the song.
def get_gamma(note: int, harmony: str) -> list:
    ans = [note]
    if harmony == "major":
        for i in major:
            note += i
            ans.append(note)

    elif harmony == "minor":
        for i in minor:
            note += i
            ans.append(note)

    for_print = []
    for x in ans:
        for_print.append(num_to_str[x % 12])

    return for_print


# Get possible keys of the song by analyzing all notes.
def get_possible_keys(mid: MidiFile) -> dict:
    notes = set(get_notes(mid, octave=False, unique=True))

    ans = {}

    for i in range(12):
        temp_gamma = get_gamma(i, "major")
        if notes.issubset(temp_gamma) or set(temp_gamma).issubset(notes):
            ans[num_to_str[i]] = temp_gamma

    for i in range(12):
        temp_gamma = get_gamma(i, "minor")
        if notes.issubset(temp_gamma) or set(temp_gamma).issubset(notes):
            ans[num_to_str[i] + "m"] = temp_gamma

    return ans


# Approach to define key of the song by last note.
def approach_last_note(mid: MidiFile, possible_keys: dict) -> list:
    notes = get_notes(mid, octave=False, unique=False)
    last_note = notes[-1]
    ans = []

    for key in possible_keys:
        stable = [possible_keys[key][0], possible_keys[key][2], possible_keys[key][4]]

        if last_note in stable:
            ans.append(key)

    if ans:
        return ans

    for key in possible_keys:
        subdominant = possible_keys[key][3]

        if last_note == subdominant:
            ans.append(key)
    return ans


# Approach to define key of the song by first note.
def approach_first_note(mid: MidiFile, possible_keys: dict) -> list:
    notes = get_notes(mid, octave=False, unique=False)
    first_note = notes[0]
    ans = []

    for key in possible_keys:
        stable = [possible_keys[key][0], possible_keys[key][2], possible_keys[key][4]]

        if first_note in stable:
            ans.append(key)

    return ans


# Approach to define key of the song by notes in level 1, 3, 4, 5
def approach_repetition_note(mid: MidiFile, possible_keys: dict) -> list:
    notes = get_notes(mid, octave=False, unique=False)
    cnt = {}

    for key in possible_keys:
        cnt[key] = 0

    for key in possible_keys:
        stable_and_subdominant = [possible_keys[key][0], possible_keys[key][2],
                                  possible_keys[key][3], possible_keys[key][4]]
        for need_note in stable_and_subdominant:
            cnt[key] += notes.count(need_note)

    max_cnt = max(cnt.values())
    ans = []
    for x in cnt:
        if max_cnt == cnt[x]:
            ans.append(x)

    return ans


# Make decision to define key by using approaches.
def get_keys(mid: MidiFile) -> list:
    possible_keys = get_possible_keys(mid)
    weight_keys = {}
    for key in possible_keys:
        weight_keys[key] = 0

    # first approach
    for key in approach_repetition_note(mid, possible_keys):
        weight_keys[key] += 0.34

    # second approach
    for key in approach_first_note(mid, possible_keys):
        weight_keys[key] += 0.33

    # third approach
    for key in approach_last_note(mid, possible_keys):
        weight_keys[key] += 0.33

    ans = max(weight_keys, key=weight_keys.get)
    return [ans, possible_keys[ans]]


class Chord:
    def __init__(self, level: int, key: str, gamma: list, inversion=False):
        self.level = level
        self.first_note = gamma[level - 1]
        self.key = key
        self.harmony = self.get_harmony()
        self.chord_group = self.get_chord_group()
        self.inversion = inversion

        info = self.get_notes()
        self.notes = info[1]
        self.type_of_chord = info[0]
        self.name = self.get_name()

    # get harmony of the chord.
    def get_harmony(self) -> str:
        if self.key[-1] == "m":
            return "minor"
        else:
            return "major"

    # get group of the chord and their weights.
    def get_chord_group(self):
        if self.level in [6, 3]:
            return ["T", 0.5]
        elif self.level == 1:
            return ["T", -1]
        elif self.level == 4:
            return ["S", 0]
        elif self.level == 2:
            return ["S", -0.5]
        elif self.level == 5:
            return ["D", 0]
        elif self.level == 7:
            return ["D", -0.5]

    # random types of chord and get it's notes.
    def get_notes(self) -> list:
        ans = []
        chords = {"major": [0, 4, 7], "minor": [0, 3, 7], "dim": [0, 3, 6], "sus2": [0, 2, 7], "sus4": [0, 5, 7]}
        possible_chords = []
        weights = []

        if self.harmony == "major":
            if self.level in [1, 5]:
                possible_chords = ["major", "sus2", "sus4"]
                weights = [0.999999, (1 - 0.999999) / 2, (1 - 0.999999) / 2]
            elif self.level in [2, 6]:
                possible_chords = ["minor", "sus2", "sus4"]
                weights = [0.999999, (1 - 0.999999) / 2, (1 - 0.999999) / 2]
            elif self.level == 3:
                possible_chords = ["minor", "sus4"]
                weights = [0.999999, 1 - 0.999999]
            elif self.level == 4:
                possible_chords = ["major", "sus2"]
                weights = [0.999999, 1 - 0.999999]
            elif self.level == 7:
                possible_chords = ["dim"]
                weights = [1]

        elif self.harmony == "minor":
            if self.level in [1, 4]:
                possible_chords = ["minor", "sus2", "sus4"]
                weights = [0.999999, (1 - 0.999999) / 2, (1 - 0.999999) / 2]
            elif self.level == 2:
                possible_chords = ["dim"]
                weights = [1]
            elif self.level in [3, 7]:
                possible_chords = ["major", "sus2", "sus4"]
                weights = [0.999999, (1 - 0.999999) / 2, (1 - 0.999999) / 2]
            elif self.level == 5:
                possible_chords = ["minor", "sus4"]
                weights = [0.999999, 1 - 0.999999]
            elif self.level == 6:
                possible_chords = ["major", "sus2"]
                weights = [0.999999, 1 - 0.999999]

        chord = random.choices(population=possible_chords, weights=weights, k=1)[0]
        for x in chords[chord]:
            ans.append((str_to_num[self.first_note] + x))

        if self.inversion:
            inversion_up = random.randint(1, 2)
            if inversion_up == 1:   # first inversion
                ans[0] += 12
            else:                   # second inversion
                ans[2] -= 12

        return [chord, ans]

    # get name of the chord.
    def get_name(self) -> str:
        name = self.first_note
        if self.type_of_chord == "major":
            return name
        else:
            if self.type_of_chord == "minor":
                return name + "m"
            else:
                return name + self.type_of_chord


class Individual:
    def __init__(self, key: str, gamma: list, num_of_chords: int):
        self.key = key
        self.gamma = gamma
        self.num_of_chords = num_of_chords
        self.values = self.create_chords()
        self.fitness_value = 0

    # Create list of chords(genes).
    def create_chords(self) -> list:
        return list([Chord(random.randint(1, 7), self.key, self.gamma) for _ in range(self.num_of_chords)])


# Create individual.
def individual_creator(key: str, gamma: list, num_of_chords: int) -> Individual:
    return Individual(key, gamma, num_of_chords)


# Create population.
def population_creator(key: str, gamma: list, num_of_chords: int, n=0) -> list:
    return list([individual_creator(key, gamma, num_of_chords) for _ in range(n)])


# Count weight for gene.
def count_fitness_one_chord(prev_chord: Chord,
                            chord: Chord, notes_in_strong_beats: list, last_chord=False) -> int:
    fitness = 0

    if prev_chord is None:
        if chord.level == 1:
            fitness += 4
    else:
        if prev_chord.chord_group[0] == "D" and chord.chord_group[0] == "S":
            fitness -= 100
        elif prev_chord.chord_group[0] == chord.chord_group[0]:
            if "sus" in prev_chord.type_of_chord and chord.level != prev_chord.level:
                fitness -= 20

            if prev_chord.notes == chord.notes:
                fitness -= 2
            elif prev_chord.chord_group[1] > chord.chord_group[1]:
                fitness += 2
        elif prev_chord.chord_group[0] == "T" and chord.chord_group[0] == "S":
            fitness += 3
        elif prev_chord.chord_group[0] == "S" and chord.chord_group[0] == "D":
            fitness += 3
        elif prev_chord.chord_group[0] == "D" and chord.chord_group[0] == "T":
            fitness += 3

        if last_chord:
            if chord.chord_group[0] == "T":
                fitness += 1

        for note in chord.notes:
            if note in prev_chord.notes:
                fitness -= 0.1

    for note in chord.notes:
        if note % 12 in map(lambda y: y % 12, notes_in_strong_beats):
            fitness += 1.5

    return fitness


# Count fitness value.
def get_fitness_individual(individual: Individual, strong_beats_lst: list) -> int:
    ans = []
    values = individual.values
    for i in range(len(values)):
        if i == 0:
            ans.append(count_fitness_one_chord(None, values[i], strong_beats_lst[i]))
        elif i == len(values) - 1:
            ans.append(count_fitness_one_chord(values[i - 1], values[i], strong_beats_lst[i], last_chord=True))
        else:
            ans.append(count_fitness_one_chord(values[i - 1], values[i], strong_beats_lst[i]))

    return sum(ans)


# Get list of fitness values for every individuals.
def get_fitness_all(population_lst: list, strong_beats_lst: list) -> list:
    ans = []
    for ind in population_lst:
        ans.append(get_fitness_individual(ind, strong_beats_lst))

    return ans


# Update fitness values of individuals in population.
def update_fitness(population_lst: list, fitness_values_lst: list):
    for i in range(len(population_lst)):
        population_lst[i].fitness_value = fitness_values_lst[i]


# Select function: Tournament method with 3 individuals.
def tournament_select(population_lst: list, p_len: int):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(max([population_lst[i1], population_lst[i2], population_lst[i3]],
                             key=lambda ind: ind.fitness_value))

    return offspring


# Crossover function: single-point crossing.
def crossover(child1: Individual, child2: Individual):
    values1 = child1.values
    values2 = child2.values
    s = random.randint(2, len(values1)-3)
    values1[s:], values2[s:] = values2[s:], values1[s:]


# Mutation function: random new gene.
def mutation(individual: Individual, key: str, gamma: list, pb=0.01):
    for i in range(len(individual.values)):
        if random.random() < pb:
            level = random.randint(1, 7)
            if level in [1, 4, 5]:
                if random.choice([True, False]) is True:    # use inversion or not
                    individual.values[i] = Chord(level, key, gamma, True)
                else:
                    individual.values[i] = Chord(level, key, gamma)
            else:
                individual.values[i] = Chord(level, key, gamma)


# Write best individual into midi file with main melody.
def write_chords(mid: MidiFile, best_individual: Individual):
    tick = mid.ticks_per_beat
    write_to = MidiTrack()

    for j in range(len(best_individual.values)):
        notes = best_individual.values[j]

        octave_chord = get_octave_of_first_note(mid)

        for note in notes.notes:
            write_to.append(Message("note_on", channel=0, note=note + octave_chord * 12, velocity=50, time=0))
        for i in range(len(notes.notes)):
            if i == 0:
                write_to.append(Message("note_off", channel=0, note=notes.notes[i] + octave_chord * 12,
                                        velocity=0, time=2 * tick))
            else:
                write_to.append(Message("note_off", channel=0, note=notes.notes[i] + octave_chord * 12,
                                        velocity=0, time=0))

    ans = MidiFile(ticks_per_beat=tick)
    ans.tracks.append(merge_tracks([mid.tracks[1], write_to]))
    ans.save(path[:-4] + "_evolution_algorithm.mid")


# Evolution process.
def evolution(input_midi: MidiFile) -> Individual:
    info_midi = get_keys(input_midi)
    print("key of the song -", info_midi[0])
    strong_beats = get_notes(input_midi, octave=False, num=True, by_strong_beat=True)
    generation_counter = 0
    max_fitness_values = []
    mean_fitness_values = []
    population = population_creator(info_midi[0], info_midi[1], len(strong_beats), n=POPULATION_SIZE)

    fitness_values = get_fitness_all(population, strong_beats)
    update_fitness(population, fitness_values)

    max_fitness_value_in_evolution = 0
    best_index = 0

    while generation_counter < MAX_GENERATION:
        generation_counter += 1
        offspring = tournament_select(population, len(population))
        offspring = list(map(deepcopy, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                crossover(child1, child2)

        for mutant in offspring:
            if random.random() < P_MUTATION:
                mutation(mutant, info_midi[0], info_midi[1], pb=1.0 / len(strong_beats))

        fresh_fitness_values = get_fitness_all(offspring, strong_beats)
        update_fitness(offspring, fresh_fitness_values)

        population[:] = offspring
        fitness_values = [ind.fitness_value for ind in population]

        max_fitness = max(fitness_values)
        mean_fitness = sum(fitness_values) / len(population)
        max_fitness_values.append(max_fitness)
        mean_fitness_values.append(mean_fitness)
        print(f"Generation {generation_counter}: Max fitness value = {max_fitness},"
              f" Mean fitness value = {mean_fitness}")

        if max_fitness >= max_fitness_value_in_evolution:
            best_index = fitness_values.index(max(fitness_values))

    return population[best_index]


if __name__ == '__main__':
    # Write name of midi file here. It should be in the same directory as this python file
    input_file = MidiFile(path, clip=True)

    best = evolution(input_file)

    print()
    print("Chords: ", end="")
    for x in best.values:
        print(x.name, end=" ")
    write_chords(input_file, best)
