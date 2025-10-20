
import random
from typing import List, Tuple

class CityDataset:
    """Manages the predefined city list."""
    
    def __init__(self, cities, seed: int = 42):
        self.cities = cities
        self.seed = seed
        random.seed(seed)
    
    def get_all_cities(self) -> List[str]:
        """Get all cities."""
        return self.cities.copy()
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2
    ) -> Tuple[List[str], List[str]]:
        """
        Split cities into train and test sets.
        
        Args:
            test_size: Fraction for test set (default: 0.2)
            
        Returns:
            (train_cities, test_cities)
        """
        cities_copy = self.cities.copy()
        random.shuffle(cities_copy)
        
        split_idx = int(len(cities_copy) * (1 - test_size))
        train = cities_copy[:split_idx]
        test = cities_copy[split_idx:]
        
        return train, test
    
    def sample_city(self, cities: List[str] = None) -> str:
        """Sample a random city."""
        if cities is None:
            cities = self.cities
        return random.choice(cities)
    
    def get_statistics(self) -> dict:
        """Get statistics about the dataset."""
        # Count by country
        countries = {}
        for city in self.cities:
            country = city.split(', ')[-1]
            countries[country] = countries.get(country, 0) + 1
        
        return {
            "total_cities": len(self.cities),
            "countries": len(countries),
            "cities_by_country": countries,
            "top_countries": sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5]
        }