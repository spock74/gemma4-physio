import random
import json
from pathlib import Path
from typing import Tuple, List, Dict, Set, Generator

def load_and_stratify_popqa(json_path: Path, popularity_threshold: int = 100000) -> Tuple[List[Dict], List[Dict]]:
    """
    Carrega o dataset PopQA e estratifica em entidades conhecidas (popularidade alta)
    e desconhecidas (popularidade baixa).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    known_entities = []
    unknown_entities = []
    
    for item in data:
        # 'wikipedia_views' é o metadado padrão do PopQA
        views = item.get("wikipedia_views", 0)
        # fallback to 0 if views is None
        if views is None:
            views = 0
            
        if views >= popularity_threshold:
            known_entities.append(item)
        elif views < 100:  # Entidades extremamente raras
            unknown_entities.append(item)
            
    return known_entities, unknown_entities

class PopQASampler:
    def __init__(self, json_path: Path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Agrupa os itens do dataset por suas classes semânticas reais
        # PopQA armazena a classe/relação no campo "triplet" ou "relation"
        self.class_map: Dict[str, List[Dict]] = {}
        for item in self.data:
            relation = item.get("relation", "unspecified")
            if relation not in self.class_map:
                self.class_map[relation] = []
            self.class_map[relation].append(item)
            
        self.all_classes: Set[str] = set(self.class_map.keys())

    def get_similar_classes_population(self, target_classes: List[str]) -> Dict[str, List[Dict]]:
        """
        Retorna a população inteira para as 5 classes similares selecionadas sem sorteio.
        """
        assert len(target_classes) == 5, "Devem ser selecionadas exatamente 5 classes similares."
        for c in target_classes:
            if c not in self.all_classes:
                raise ValueError(f"Classe semântica '{c}' não encontrada no dataset.")
        return {c: self.class_map[c] for c in target_classes}

    def sample_5x5_representatives(self, active_classes: List[str], seed: int = 42) -> Dict[str, List[Dict]]:
        """
        Extrai exatamente 5 representantes aleatórios de cada uma das 5 classes selecionadas.
        """
        rng = random.Random(seed)
        sampled_data = {}
        for c in active_classes:
            population = self.class_map[c]
            # Seleciona 5 representantes sem reposição dentro da própria classe
            sampled_data[c] = rng.sample(population, min(5, len(population)))
        return sampled_data

    def generate_random_subsets_without_replacement(self) -> Generator[List[str], None, None]:
        """
        Gerador que sorteia subconjuntos de 5 classes sem reposição até esgotar o pool do benchmark.
        """
        available_classes = list(self.all_classes)
        random.shuffle(available_classes)
        
        while len(available_classes) >= 5:
            # Retira 5 classes sem reposição
            subset = [available_classes.pop() for _ in range(5)]
            yield subset
