import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ParsedQuery:
    raw_query: str
    items: List[Dict[str, str]]  # [{'type': 'shirt', 'color': 'blue'}, ...]
    scene: Optional[str]
    location: Optional[str]
    style: Optional[str]
    action: Optional[str]
    relation: Optional[str]
    attributes: Dict[str, any]


class QueryParser:

    # Fashion garment keywords
    GARMENTS = [
        'shirt', 'blouse', 'top', 't-shirt', 'tshirt', 'tee',
        'pants', 'trousers', 'jeans', 'slacks',
        'dress', 'gown', 'frock',
        'jacket', 'blazer', 'coat', 'cardigan',
        'suit', 'tuxedo',
        'tie', 'bowtie', 'necktie',
        'skirt', 'shorts',
        'sweater', 'pullover', 'hoodie', 'sweatshirt',
        'shoes', 'boots', 'sneakers', 'heels', 'sandals',
        'hat', 'cap', 'beanie',
        'scarf', 'shawl',
        'bag', 'purse', 'backpack',
        'belt', 'watch', 'glasses', 'sunglasses',
        'raincoat', 'poncho',
        'outfit', 'attire', 'clothing', 'wear'
    ]
    
    # Color keywords with variations
    COLORS = {
        'red': ['red', 'crimson', 'scarlet', 'ruby'],
        'blue': ['blue', 'navy', 'azure', 'cobalt', 'sapphire'],
        'green': ['green', 'emerald', 'olive', 'lime'],
        'yellow': ['yellow', 'golden', 'amber'],
        'orange': ['orange', 'tangerine'],
        'purple': ['purple', 'violet', 'lavender', 'plum'],
        'pink': ['pink', 'rose', 'magenta', 'fuchsia'],
        'white': ['white', 'ivory', 'cream', 'beige'],
        'black': ['black', 'ebony'],
        'gray': ['gray', 'grey', 'silver', 'charcoal'],
        'brown': ['brown', 'tan', 'khaki', 'chocolate'],
    }
    
    # Brightness/intensity modifiers
    COLOR_MODIFIERS = [
        'bright', 'dark', 'light', 'pale', 'deep', 'vivid',
        'pastel', 'neon', 'muted', 'bold'
    ]
    
    # Scene/location keywords
    SCENES = {
        'office': ['office', 'workplace', 'corporate'],
        'street': ['street', 'urban', 'city'],
        'park': ['park', 'garden', 'outdoor'],
        'home': ['home', 'house', 'indoor', 'living room', 'bedroom'],
        'beach': ['beach', 'seaside', 'shore'],
        'cafe': ['cafe', 'coffee shop', 'restaurant'],
        'gym': ['gym', 'fitness', 'workout'],
        'formal': ['formal setting', 'event', 'gala', 'ceremony']
    }
    
    # Style descriptors
    STYLES = {
        'casual': ['casual', 'relaxed', 'laid-back', 'everyday', 'weekend'],
        'formal': ['formal', 'professional', 'business', 'corporate', 'suit'],
        'elegant': ['elegant', 'sophisticated', 'classy', 'chic'],
        'sporty': ['sporty', 'athletic', 'active', 'gym'],
        'street': ['streetwear', 'urban', 'hip'],
        'vintage': ['vintage', 'retro', 'classic'],
        'modern': ['modern', 'contemporary', 'trendy'],
    }
    
    # Action words
    ACTIONS = [
        'sitting', 'standing', 'walking', 'running',
        'posing', 'wearing', 'holding'
    ]
    
    # Spatial relations
    RELATIONS = [
        'on', 'in', 'near', 'beside', 'next to',
        'with', 'and', 'plus'
    ]
    
    def __init__(self):
        # Build reverse lookup for colors
        self.color_lookup = {}
        for base_color, variations in self.COLORS.items():
            for var in variations:
                self.color_lookup[var] = base_color
    
    def parse(self, query: str) -> ParsedQuery:
        query_lower = query.lower()
        
        # Extract colors with modifiers
        colors = self._extract_colors(query_lower)
        
        # Extract garments
        garments = self._extract_garments(query_lower)
        
        # Match colors to garments
        items = self._match_colors_to_garments(colors, garments, query_lower)
        
        # Extract scene/location
        scene = self._extract_scene(query_lower)
        
        # Extract style
        style = self._extract_style(query_lower)
        
        # Extract actions
        action = self._extract_action(query_lower)
        
        # Extract spatial relations
        relation = self._extract_relation(query_lower)
        
        # Additional attributes
        attributes = {
            'has_person': self._contains_person_reference(query_lower),
            'brightness_modifier': self._extract_brightness(query_lower),
            'is_compositional': len(items) > 1,
            'requires_context': scene is not None or relation is not None
        }
        
        return ParsedQuery(
            raw_query=query,
            items=items,
            scene=scene,
            location=scene,  # Alias
            style=style,
            action=action,
            relation=relation,
            attributes=attributes
        )
    
    def _extract_colors(self, text: str) -> List[Dict[str, str]]:
        found_colors = []
        
        for base_color, variations in self.COLORS.items():
            for color_word in variations:
                # Check for modifier + color
                for modifier in self.COLOR_MODIFIERS:
                    pattern = f"{modifier}\\s+{color_word}"
                    if re.search(pattern, text):
                        found_colors.append({
                            'color': base_color,
                            'modifier': modifier,
                            'full': f"{modifier} {color_word}"
                        })
                        break
                else:
                    # Just the color without modifier
                    if re.search(r'\b' + color_word + r'\b', text):
                        found_colors.append({
                            'color': base_color,
                            'modifier': None,
                            'full': color_word
                        })
        
        return found_colors
    
    def _extract_garments(self, text: str) -> List[str]:
        found_garments = []
        
        for garment in self.GARMENTS:
            if re.search(r'\b' + garment + r'\b', text):
                found_garments.append(garment)
        
        return found_garments
    
    def _match_colors_to_garments(self, colors: List[Dict], garments: List[str], text: str) -> List[Dict]:
        items = []
        
        if len(colors) == 0 and len(garments) == 0:
            # No specific items mentioned
            return items
        
        if len(garments) == 0:
            # Only colors, assume generic clothing
            for color_info in colors:
                items.append({
                    'type': 'clothing',
                    'color': color_info['color'],
                    'color_modifier': color_info['modifier']
                })
        
        elif len(colors) == 0:
            # Only garments, no color specified
            for garment in garments:
                items.append({
                    'type': garment,
                    'color': None,
                    'color_modifier': None
                })
        
        elif len(colors) == 1 and len(garments) == 1:
            # Simple case: one color, one garment
            items.append({
                'type': garments[0],
                'color': colors[0]['color'],
                'color_modifier': colors[0]['modifier']
            })
        
        else:
            # Complex case: multiple colors and/or garments
            # Try to match by proximity in text
            for garment in garments:
                # Find closest color
                garment_pos = text.find(garment)
                
                closest_color = None
                min_distance = float('inf')
                
                for color_info in colors:
                    color_pos = text.find(color_info['full'])
                    distance = abs(color_pos - garment_pos)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = color_info
                
                items.append({
                    'type': garment,
                    'color': closest_color['color'] if closest_color and min_distance < 50 else None,
                    'color_modifier': closest_color['modifier'] if closest_color and min_distance < 50 else None
                })
        
        return items
    
    def _extract_scene(self, text: str) -> Optional[str]:
        for scene_type, keywords in self.SCENES.items():
            for keyword in keywords:
                if keyword in text:
                    return scene_type
        return None
    
    def _extract_style(self, text: str) -> Optional[str]:
        for style_type, keywords in self.STYLES.items():
            for keyword in keywords:
                if keyword in text:
                    return style_type
        return None
    
    def _extract_action(self, text: str) -> Optional[str]:
        for action in self.ACTIONS:
            if action in text:
                return action
        return None
    
    def _extract_relation(self, text: str) -> Optional[str]:
        # Look for patterns like "X and Y" or "X with Y"
        if ' and ' in text or ' with ' in text or ' plus ' in text:
            return 'compositional'
        
        for relation in self.RELATIONS:
            if relation in text:
                return relation
        
        return None
    
    def _contains_person_reference(self, text: str) -> bool:
        person_keywords = ['person', 'someone', 'man', 'woman', 'people', 'individual']
        return any(keyword in text for keyword in person_keywords)
    
    def _extract_brightness(self, text: str) -> Optional[str]:
        for modifier in self.COLOR_MODIFIERS:
            if modifier in text:
                return modifier
        return None
    
    def generate_search_phrases(self, parsed: ParsedQuery) -> List[str]:
        phrases = [parsed.raw_query]  # Original query
        
        # Generate item-specific phrases
        for item in parsed.items:
            parts = []
            
            if item['color_modifier']:
                parts.append(item['color_modifier'])
            if item['color']:
                parts.append(item['color'])
            if item['type']:
                parts.append(item['type'])
            
            if parts:
                phrases.append(' '.join(parts))
        
        # Add style phrase
        if parsed.style:
            phrases.append(f"{parsed.style} outfit")
            phrases.append(f"{parsed.style} fashion")
        
        # Add scene phrase
        if parsed.scene:
            phrases.append(f"photo in {parsed.scene}")
            phrases.append(f"{parsed.scene} setting")
        
        # Compositional phrase
        if len(parsed.items) > 1:
            item_descriptions = []
            for item in parsed.items:
                desc = []
                if item['color']:
                    desc.append(item['color'])
                if item['type']:
                    desc.append(item['type'])
                if desc:
                    item_descriptions.append(' '.join(desc))
            
            if len(item_descriptions) > 1:
                phrases.append(' and '.join(item_descriptions))
        
        return phrases


def test_parser():
    parser = QueryParser()
    
    test_queries = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        parsed = parser.parse(query)
        print(f"Items: {parsed.items}")
        print(f"Scene: {parsed.scene}")
        print(f"Style: {parsed.style}")
        print(f"Action: {parsed.action}")
        print(f"Relation: {parsed.relation}")
        print(f"Attributes: {parsed.attributes}")
        print(f"Search phrases: {parser.generate_search_phrases(parsed)}")


if __name__ == "__main__":
    test_parser()