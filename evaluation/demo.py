import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from retriever.search_engine import MultiVectorRetriever
from configs.config import Config
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_results(query: str, results: list, num_display: int = 10):
    num_display = min(num_display, len(results))
    
    # Create figure
    cols = 5
    rows = (num_display + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle(f'Query: "{query}"', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        if idx < len(results):
            result = results[idx]
            
            # Load and display image
            try:
                img = Image.open(result['image_path'])
                ax.imshow(img)
                
                # Add title with rank and score
                title = f"#{idx+1} (score: {result['score']:.3f})"
                ax.set_title(title, fontsize=10)
                
                # Add metadata
                scene = result['metadata'].get('scene', 'unknown')
                num_items = result['metadata'].get('num_items', 0)
                info_text = f"Scene: {scene}\nItems: {num_items}"
                ax.text(0.02, 0.98, info_text, 
                       transform=ax.transAxes,
                       fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{e}", 
                       ha='center', va='center')
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def interactive_demo():
    print("Interactive Demo")
    
    # Initialize retriever
    config = Config()
    retriever = MultiVectorRetriever(config=config)
    
    print("\n-> System ready! Enter queries to search.")
    print("Commands:")
    print("  - Type a query to search")
    print("  - Type 'examples' to see example queries")
    print("  - Type 'quit' to exit")
    
    while True:
        print("\n" + "-" * 80)
        query = input("Enter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if query.lower() == 'examples':
            print("\nExample queries:")
            for i, example in enumerate(config.EVALUATION_QUERIES, 1):
                print(f"{i}. {example}")
            continue
        
        if not query:
            continue
        
        # Retrieve
        try:
            results = retriever.retrieve(query, top_k=10)
            
            # Display text results
            print(f"\nTop 10 results for: \"{query}\"")
            print(f"{'Rank':<6} {'Filename':<50} {'Score':<8}")
            print("-" * 70)
            
            for i, result in enumerate(results, 1):
                filename = Path(result['image_path']).name
                score = result['score']
                print(f"{i:<6} {filename:<50} {score:<8.4f}")
            
            # Ask if user wants to visualize
            show_vis = input("\nShow visual results? (y/n): ").strip().lower()
            if show_vis == 'y':
                visualize_results(query, results, num_display=10)
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def batch_demo():
    print("Batch Demo- All Evaluation queries")
    
    config = Config()
    retriever = MultiVectorRetriever(config=config)
    
    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, query in enumerate(config.EVALUATION_QUERIES, 1):
        print(f"\n[{i}/{len(config.EVALUATION_QUERIES)}] Processing: {query}")
        
        results = retriever.retrieve(query, top_k=10)
        
        # Save visualization
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(f'Query {i}: "{query}"', fontsize=14, fontweight='bold')
        
        for j, result in enumerate(results[:10], 1):
            ax = plt.subplot(2, 5, j)
            
            try:
                img = Image.open(result['image_path'])
                ax.imshow(img)
                ax.set_title(f"#{j}\nScore: {result['score']:.3f}", fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center')
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"query_{i}_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  -> Saved visualization to {output_dir / f'query_{i}_results.png'}")
    
    print(f"\n-> All results saved to: {output_dir}")


def quick_test():
    print("Quick Test")
    
    config = Config()
    retriever = MultiVectorRetriever(config=config)
    
    # Test query
    query = "A person in a bright yellow raincoat"
    print(f"\nTest query: {query}")
    
    results = retriever.retrieve(query, top_k=5)
    
    print(f"\nTop 5 results:")
    for i, result in enumerate(results, 1):
        filename = Path(result['image_path']).name
        print(f"{i}. {filename} (score: {result['score']:.3f})")
    
    # Show visualization
    visualize_results(query, results, num_display=5)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fashion Retrieval Demo')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'batch', 'quick'],
                       help='Demo mode')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_demo()
    elif args.mode == 'batch':
        batch_demo()
    elif args.mode == 'quick':
        quick_test()


if __name__ == "__main__":
    main()