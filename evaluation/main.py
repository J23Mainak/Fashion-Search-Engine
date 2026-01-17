import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config


def main():
    parser = argparse.ArgumentParser(
        description='Fashion Retrieval System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Index dataset
        python main.py index --data_dir /path/to/images
        
        # Query system
        python main.py query "red tie and white shirt"
        
        # Run evaluation
        python main.py evaluate
        
        # Interactive demo
        python main.py demo
            """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build index from images')
    index_parser.add_argument('--data_dir', type=str, help='Path to image directory')
    index_parser.add_argument('--output_dir', type=str, help='Path to save indices')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query_text', type=str, help='Natural language query')
    query_parser.add_argument('--top_k', type=int, default=10, help='Number of results')
    query_parser.add_argument('--visualize', action='store_true', help='Show visual results')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--mode', type=str, default='interactive',
                            choices=['interactive', 'batch', 'quick'])
    
    args = parser.parse_args()
    
    if args.command == 'index':
        print("Indexing Dataset...")
        
        from indexer.build_index import MultiVectorIndexer
        
        config = Config()
        if args.data_dir:
            config.IMAGES_DIR = Path(args.data_dir)
        if args.output_dir:
            config.INDEX_DIR = Path(args.output_dir)
        
        indexer = MultiVectorIndexer(config)
        indexer.index_dataset(str(config.IMAGES_DIR))
        indexer.build_faiss_indices()
        indexer.save(args.output_dir)
        
        print("\n-> Indexing complete!")
        
    elif args.command == 'query':
        print("Querying System...")
        
        from retriever.search_engine import MultiVectorRetriever
        
        retriever = MultiVectorRetriever()
        results = retriever.retrieve(args.query_text, top_k=args.top_k)
        
        print(f"\nTop {len(results)} results for: \"{args.query_text}\"")
        
        for i, result in enumerate(results, 1):
            filename = Path(result['image_path']).name
            score = result['score']
            print(f"{i:2d}. {filename:<50} Score: {score:.4f}")
        
        if args.visualize:
            from demo import visualize_results
            visualize_results(args.query_text, results, num_display=min(10, len(results)))
        
    elif args.command == 'evaluate':
        print("Evaluating System...")
        
        from evaluation.evaluate import evaluate_system
        evaluate_system()
        
    elif args.command == 'demo':
        from demo import interactive_demo, batch_demo, quick_test
        
        if args.mode == 'interactive':
            interactive_demo()
        elif args.mode == 'batch':
            batch_demo()
        elif args.mode == 'quick':
            quick_test()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()