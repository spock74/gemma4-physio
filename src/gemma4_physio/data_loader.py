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
        assert len(target_classes) >= 1, "Devem ser selecionadas classes similares."
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

    def sample_purified_representatives(self, active_classes: List[str], seed: int = 42, count_per_class: int = 5) -> Dict[str, List[Dict]]:
        """
        Extrai representantes aleatórios de cada classe semântica, mas purifica a amostragem
        removendo atalhos nominais, vazamentos ortográficos (não-ASCII) e cópias diretas.
        """
        import json
        rng = random.Random(seed)
        sampled_data = {}
        
        for c in active_classes:
            population = self.class_map[c]
            purified_population = []
            
            for item in population:
                question = (item.get("question") or "").lower()
                subject = (item.get("subject") or "").lower()
                
                # Parse answers
                try:
                    raw_answer = item.get('answer') or '[]'
                    answers = json.loads(raw_answer)
                    if not isinstance(answers, list):
                        answers = [str(answers)]
                except Exception:
                    answers = [str(item.get('answer') or '')]
                    
                # 1. Filtro contra cópias nominais (se a resposta estiver contida no prompt ou no sujeito)
                has_copy_shortcut = False
                for ans in answers:
                    ans_clean = ans.strip().lower()
                    if ans_clean in question or ans_clean in subject:
                        has_copy_shortcut = True
                        break
                if has_copy_shortcut:
                    continue
                    
                # 2. Filtro contra vazamentos ortográficos (só aceita caracteres ASCII puros no sujeito e respostas)
                # Isso remove nomes poloneses com "ę", finlandeses com "ä"/"ö", etc.
                try:
                    subject.encode('ascii')
                    for ans in answers:
                        ans.encode('ascii')
                except UnicodeEncodeError:
                    # Contém caracteres não-ASCII
                    continue
                    
                purified_population.append(item)
                
            # Se a classe purificada ficou muito pequena, faz fallback para a população original
            if len(purified_population) < count_per_class:
                purified_population = population
                
            sampled_data[c] = rng.sample(purified_population, min(count_per_class, len(purified_population)))
            
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
