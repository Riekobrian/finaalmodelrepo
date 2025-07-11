import numpy as np
from typing import List, Dict, Tuple, Union, Any

def analyze_feature_sensitivity(input_dict: Dict[str, Any], feature: str, predict_fn) -> Tuple[List[float], List[float]]:
    """Analyze how changes in a numeric feature affect predictions"""
    base_value = input_dict[feature]
    variations = []
    predictions = []
    
    try:
        # Create variations based on feature type
        if 'age' in feature.lower():
            variations = list(range(max(0, int(base_value - 5)), int(base_value + 6)))
        elif 'mileage' in feature.lower():
            step = 10000
            variations = list(range(max(0, int(base_value - 5*step)), int(base_value + 6*step), step))
        elif 'engine' in feature.lower():
            step = 200
            variations = list(range(max(600, int(base_value - 3*step)), int(base_value + 4*step), step))
        elif 'power' in feature.lower() or 'torque' in feature.lower():
            step = 20
            variations = list(range(max(30, int(base_value - 3*step)), int(base_value + 4*step), step))
        else:
            # Default variation: ±30% in 6 steps
            step = base_value * 0.1
            variations = [base_value + i*step for i in range(-3, 4)]
        
        # Get predictions for each variation
        for value in variations:
            input_copy = input_dict.copy()
            input_copy[feature] = value
            try:
                pred = predict_fn(input_copy)
                predictions.append(pred)
            except:
                predictions.append(None)
                
    except Exception as e:
        print(f"Error in sensitivity analysis: {str(e)}")
        return [], []
        
    return variations, predictions

def get_feature_impact(old_input: Dict[str, Any], new_input: Dict[str, Any], predict_fn) -> Tuple[List[Dict[str, Any]], float, float]:
    """Analyze the impact of feature changes on the prediction"""
    try:
        # Get predictions
        old_price = predict_fn(old_input)
        new_price = predict_fn(new_input)
        
        # Compare features
        changes = []
        for key in old_input:
            if key in new_input and old_input[key] != new_input[key]:
                # Try isolated impact
                test_input = old_input.copy()
                test_input[key] = new_input[key]
                test_price = predict_fn(test_input)
                
                impact = test_price - old_price
                percent = (impact / old_price) * 100
                
                changes.append({
                    'feature': key,
                    'old_value': old_input[key],
                    'new_value': new_input[key],
                    'impact': impact,
                    'percent': percent
                })
        
        # Sort by absolute impact
        changes.sort(key=lambda x: abs(x['impact']), reverse=True)
        return changes, old_price, new_price
        
    except Exception as e:
        print(f"Error in impact analysis: {str(e)}")
        return [], 0, 0

def format_impact(changes: List[Dict[str, Any]]) -> str:
    """Format impact analysis results as markdown"""
    if not changes:
        return "No significant changes detected"
        
    lines = []
    for change in changes:
        feature = change['feature'].replace('_', ' ').title()
        if abs(change['percent']) >= 0.1:  # Only show meaningful changes
            direction = "increased" if change['impact'] > 0 else "decreased"
            lines.append(
                f"- {feature}: Changed from {change['old_value']} to {change['new_value']}, "
                f"{direction} price by {abs(change['percent']):.1f}%"
            )
    
    return "\n".join(lines) if lines else "No significant price impact detected"
