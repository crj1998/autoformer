
import random
from typing import Optional, Dict

class AttrDict(dict):
    MARKER = object()

    def __init__(self, data=None):
        if data is None: return
        assert isinstance(data, dict), 'expected dict'
        for key in data:
            self.__setitem__(key, data[key])

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, AttrDict.MARKER)
        if found is AttrDict.MARKER:
            found = AttrDict()
            super(AttrDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


class Individual(AttrDict):
    def __init__(self, data):
        super().__init__(data)

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


class EvolutionSearcher:
    MAX_RETRY = 16
    def __init__(self, metric_func, 
        search_space: Dict[str, list], 
        evolution_iters: int = 10, 
        population_num: int = 50, 
        parent_num: int = 20,
        mutation_prob: float = 0.5,
        mutation_num: int = 25,
        crossover_num: int = 25,
        optimize: str = "max",
    ):
        assert optimize in ['max', 'min']
        self.metric_func = metric_func
        self.search_space = search_space
        self.evolution_iters = evolution_iters
        self.population_num = population_num
        self.parent_num = parent_num
        self.mutation_prob = mutation_prob
        self.mutation_num = mutation_num
        self.crossover_num = crossover_num
        self.optimize = optimize

        self.searchable = tuple(search_space.keys())
        self._cache = dict()

    def _cache_hit(self, config):
        if config in self._cache:
            return True
        else:
            self._cache[config] = None
            return False

    def resample(self):
        for _ in range(self.MAX_RETRY):
            config = Individual({k: random.choice(v) for k, v in self.search_space.items()})
            if not self._cache_hit(config): break
        return config

    def random(self, population):
        while len(population) < self.population_num:
            for i in range(self.MAX_RETRY):
                config = Individual({k: random.choice(v) for k, v in self.search_space.items()})
                if not self._cache_hit(config): break
            self._cache[config] = self.metric_func(dict(config))
            population.append(config)
            if i == self.MAX_RETRY - 1:
                break
        return population

    def mutation(self, parents):
        res = []
        while len(res) < self.mutation_num:
            for i in range(self.MAX_RETRY):
                config = random.choice(parents)
                config = Individual({
                    k: random.choice(self.search_space[k]) if random.random() < self.mutation_prob else v
                    for k, v in config.items()
                })
                if not self._cache_hit(config): break

            self._cache[config] = self.metric_func(dict(config))
            res.append(config)
            if i == self.MAX_RETRY - 1:
                break
        return res


    def crossover(self, parents):
        res = []
        while len(res) < self.crossover_num:
            for i in range(self.MAX_RETRY):
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                if p1 == p2: continue
                config = Individual({k: random.choice([p1[k], p2[k]]) for k in self.searchable})
                if not self._cache_hit(config):
                    self._cache[config] = self.metric_func(dict(config))
                    res.append(config)
                    break
            if i == self.MAX_RETRY - 1:
                break
        return res

    def run(self, init=None, topk=1):
        population = []
        if init is not None:
            for p in init:
                config = Individual(p)
                self._cache[config] = self.metric_func(dict(config))
                population.append(config)
        self.random(population)
        population = sorted(population, key=lambda x: self._cache[x]['metric'], reverse=(self.optimize == 'max'))[:self.population_num]
        for _ in range(1, self.evolution_iters):
            parents = sorted(population, key=lambda x: self._cache[x]['metric'], reverse=(self.optimize == 'max'))[:self.parent_num]
            mutation = self.mutation(parents)
            crossover = self.crossover(parents)
            population = population + mutation + crossover
            self.random(population)
            population = sorted(population, key=lambda x: self._cache[x]['metric'], reverse=(self.optimize == 'max'))[:self.population_num]
        return {p: self._cache[p] for p in population[:max(topk, 1)]}


if __name__ == '__main__':
    search_space = {
        f"layer_{layer}": [1, 2, 4, 8, 16] for layer in range(12)
    }
    def f(kwargs):
        return {'metric': kwargs["layer_1"] + kwargs["layer_3"]}
    searcher = EvolutionSearcher(f, search_space, optimize='max')
    searcher.run()