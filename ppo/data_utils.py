
import random
from typing import List, Tuple

CITIES = [
    'Guayaquil, Ecuador', 'Taipei, China', 'Zibo, China', 'Jinan, China', 'Alexandria, Egypt',
    'Berlin, Germany', 'Sydney, Australia', 'Istanbul, Turkey', 'Osaka, Japan', 'Hong Kong, China',
    'Bogota, Colombia', 'Jakarta, Indonesia', 'Bogor, Indonesia', 'Bandung, Indonesia', 'Kolkata, India',
    'Tashkent, Uzbekistan', 'Chengdu, China', 'Giza, Egypt', 'Semarang, Indonesia', 'Lima, Peru',
    'Hyderabad, India', 'Havana, Cuba', 'Harbin, China', 'Izmir, Turkey', 'Brasilia, Brazil',
    'Shenyang, China', 'Delhi, India', 'Baghdad, Iraq', 'Rio de Janeiro, Brazil', 'London, UK',
    'Rome, Italy', 'Los Angeles, USA', 'Mexico City, Mexico', 'Bucharest, Romania', 'Ho Chi Minh City, Vietnam',
    'Daegu, South Korea', 'Toronto, Canada', 'Surabaya, Indonesia', 'Bangalore, India', 'Fortaleza, Brazil',
    'Yokohama, Japan', 'Salvador, Brazil', 'St. Petersburg, Russia', 'Beijing, China', 'Wuhan, China',
    'Karachi, Pakistan', 'Cirebon, Indonesia', 'Dhaka, Bangladesh', 'Chicago, USA', 'Mumbai, India',
    'Guangzhou, China', 'Santiago, Chile', 'Budapest, Hungary', 'Tehran, Iran', 'Houston, USA',
    'Casablanca, Morocco', 'Kinshasa, Congo', 'Malang, Indonesia', 'Qingdao, China', "Xi'an, China",
    'Caracas, Venezuela', 'Abidjan, Côte d\'Ivoire', 'Medellin, Colombia', 'Tokyo, Japan', 'Chennai, India',
    'Kanpur, India', 'Bangkok, Thailand', 'Addis Ababa, Ethiopia', 'Busan, South Korea', 'Dalian, China',
    'Tianjin, China', 'Mashhad, Iran', 'Yangon, Myanmar', 'Sukabumi, Indonesia', 'Moscow, Russia',
    'Incheon, South Korea', 'Buenos Aires, Argentina', 'Cali, Colombia', 'New York, USA', 'Lahore, Pakistan',
    'Ahmedabad, India', 'Chongqing, China', 'Changchun, China', 'Nanjing, China', 'Madrid, Spain',
    'Taiyuan, China', 'Shanghai, China', 'Cairo, Egypt', 'Medan, Indonesia', 'Belo Horizonte, Brazil',
    'Paris, France', 'Nagoya, Japan', 'São Paulo, Brazil', 'Singapore, Singapore', 'Kiev, Ukraine',
    'Pyongyang, North Korea', 'Faisalabad, Pakistan', 'Ankara, Turkey', 'Quezon City, Philippines'
]


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