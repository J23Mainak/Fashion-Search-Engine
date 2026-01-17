import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from retriever.search_engine import MultiVectorRetriever
from configs.config import Config
import json
from datetime import datetime


def evaluate_system():
    print("\nEvaluation")
    
    # Initialize
    config = Config()
    retriever = MultiVectorRetriever(config=config)
    
    # Evaluation queries from assignment
    evaluation_queries = [
        {
            'id': 1,
            'type': 'Attribute Specific',
            'query': 'A person in a bright yellow raincoat',
            'expected_features': ['yellow', 'bright', 'raincoat', 'person']
        },
        {
            'id': 2,
            'type': 'Contextual/Place',
            'query': 'Professional business attire inside a modern office',
            'expected_features': ['professional', 'business', 'office', 'formal']
        },
        {
            'id': 3,
            'type': 'Complex Semantic',
            'query': 'Someone wearing a blue shirt sitting on a park bench',
            'expected_features': ['blue', 'shirt', 'park', 'bench', 'sitting']
        },
        {
            'id': 4,
            'type': 'Style Inference',
            'query': 'Casual weekend outfit for a city walk',
            'expected_features': ['casual', 'weekend', 'city', 'urban']
        },
        {
            'id': 5,
            'type': 'Compositional',
            'query': 'A red tie and a white shirt in a formal setting',
            'expected_features': ['red', 'tie', 'white', 'shirt', 'formal']
        }
    ]
    
    # Store all results
    all_results = []
    
    # Evaluate each query
    for eval_query in evaluation_queries:
        print(f"Query {eval_query['id']}: {eval_query['type']}")
        print(f"Query: {eval_query['query']}")
        print(f"Expected features: {', '.join(eval_query['expected_features'])}")
        
        # Retrieve
        try:
            results = retriever.retrieve(eval_query['query'], top_k=10)
            
            # Display top 10 results
            print(f"\nTop 10 Results:")
            print(f"{'Rank':<6} {'Filename':<50} {'Score':<8} {'Scene':<12} {'Items'}")
            
            for i, result in enumerate(results, 1):
                filename = Path(result['image_path']).name
                score = result['score']
                scene = result['metadata'].get('scene', 'unknown')
                num_items = result['metadata'].get('num_items', 0)
                
                print(f"{i:<6} {filename:<50} {score:<8.4f} {scene:<12} {num_items}")
            
            # Store results
            query_result = {
                'query_id': eval_query['id'],
                'query_type': eval_query['type'],
                'query_text': eval_query['query'],
                'top_10_results': [
                    {
                        'rank': i,
                        'image_path': result['image_path'],
                        'score': float(result['score']),
                        'metadata': result['metadata']
                    }
                    for i, result in enumerate(results, 1)
                ]
            }
            all_results.append(query_result)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results to JSON
    output_dir = Path(config.PROJECT_ROOT) / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_queries': len(evaluation_queries),
            'results': all_results
        }, f, indent=2)
    
    print("Finish Evaluation!")
    print(f"Results saved to: {output_file}")
    
    print("-> Summary:")
    
    for i, query_result in enumerate(all_results, 1):
        avg_score = sum(r['score'] for r in query_result['top_10_results']) / 10
        print(f"Query {i} ({query_result['query_type']}):")
        print(f"  Average score (top-10): {avg_score:.4f}")
        print(f"  Best match score: {query_result['top_10_results'][0]['score']:.4f}")
    
    return all_results


def analyze_results(results_file: str = None):
    if results_file is None:
        # Find most recent results file
        results_dir = Path("evaluation/results")
        result_files = sorted(results_dir.glob("evaluation_results_*.json"))
        if not result_files:
            print("No results files found!")
            return
        results_file = result_files[-1]
    
    print(f"Analyzing results from: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("-> Detailed Analysis")
    
    for query_result in data['results']:
        print(f"\nQuery {query_result['query_id']}: {query_result['query_type']}")
        print(f"Text: {query_result['query_text']}")
        
        scores = [r['score'] for r in query_result['top_10_results']]
        
        print(f"Score statistics:")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std:  {np.std(scores):.4f}")
        print(f"  Min:  {np.min(scores):.4f}")
        print(f"  Max:  {np.max(scores):.4f}")
        
        # Count by scene
        scenes = [r['metadata']['scene'] for r in query_result['top_10_results']]
        scene_counts = {}
        for scene in scenes:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        
        print(f"\nScene distribution in top-10:")
        for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {scene}: {count}")

def main():
    import numpy as np
    
    # Run evaluation
    results = evaluate_system()
    
    print("Evaluation Complete")
    print("\nNext steps:")
    print("1. Review the retrieved images visually")
    print("2. Compute Recall@K metrics with ground truth labels")
    print("3. Analyze failure cases")
    print("4. Fine-tune model on your specific dataset")


if __name__ == "__main__":
    main()