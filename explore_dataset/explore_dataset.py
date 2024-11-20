import os
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.preprocessing import *

def analyze_folder(folder, label):
    reviews = load_reviews(folder)
    print(f"Analysis for {label} reviews:")
    print(f"- Total reviews: {len(reviews)}")
    print(f"- Average review length: {sum(len(r.split()) for r in reviews) / len(reviews):.2f} words")
    print(f"- Max review length: {max(len(r.split()) for r in reviews)} words")
    print(f"- Min review length: {min(len(r.split()) for r in reviews)} words")
    return reviews

def plot_review_lengths(reviews, label, save_path=None):
    lengths = [len(r.split()) for r in reviews]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7)
    plt.title(f"Distribution of Review Lengths ({label})")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    base_folder = "data/train/"
    pos_folder = os.path.join(base_folder, "pos")
    neg_folder = os.path.join(base_folder, "neg")
    unsup_folder = os.path.join(base_folder, "unsup")

    # Analyze positive reviews
    pos_reviews = analyze_folder(pos_folder, "Positive")
    plot_review_lengths(pos_reviews, "Positive", save_path="explore_dataset/positive_lengths.png")

    # Analyze negative reviews
    neg_reviews = analyze_folder(neg_folder, "Negative")
    plot_review_lengths(neg_reviews, "Negative", save_path="explore_dataset/negative_lengths.png")

    # Analyze unlabeled reviews (optional)
    if os.path.exists(unsup_folder):
        unsup_reviews = analyze_folder(unsup_folder, "Unlabeled")
        plot_review_lengths(unsup_reviews, "Unlabeled", save_path="explore_dataset/unlabeled_lengths.png")

    # Vocabulary stats
    all_reviews = pos_reviews + neg_reviews
    word_counter = Counter(word for review in all_reviews for word in review.split())
    print(f"Total unique words: {len(word_counter)}")
    print(f"Most common words: {word_counter.most_common(10)}")

if __name__ == "__main__":
    main()

