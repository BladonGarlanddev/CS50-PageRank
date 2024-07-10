import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    print(f"Page of focus for transition model: {page}")
    direct_links = corpus[page]
    print(f"direct links: {direct_links}")
    other_links = corpus.keys() - direct_links 
    print(f"other links: {other_links}")

    n_outgoing_links = len(corpus[page])
    
    prob_distro = {page: 0 for page in corpus}

    


    if n_outgoing_links > 0:

        lesser_distro = (1 - DAMPING) / len(corpus)
        greater_distro = DAMPING / len(direct_links)

        for link in direct_links:
            prob_distro[link] = greater_distro
        
        for link in corpus:
            prob_distro[link] += lesser_distro

        
        total = 0
        for page, probability in prob_distro.items():
            print(f"Page, Probability: ({page, probability})")
            total += probability

        print(f"total: {total} \n" )

        return prob_distro
    
    else:
        for page in corpus:
            prob_distro[page] = 1 / len(corpus)
    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    
    

    current_page = random.choice(list(corpus.keys()))  # Start from a random page
    for _ in range(n):
        pageRanks[current_page] += 1  # Increment the count for the current page

        # Determine the next page based on the transition model probabilities
        probabilities = transition_model(corpus, current_page, damping_factor)
        next_pages = list(probabilities.keys())
        next_probs = list(probabilities.values())
        current_page = random.choices(next_pages, weights=next_probs, k=1)[0]

    # Normalize the PageRank values so that they sum to 1
    total = sum(pageRanks.values())
    for page in pageRanks:
        pageRanks[page] /= total

    return pageRanks
    """

    print(f"corpus: {corpus} \n\n\n")

    pageRanks = {page: 0 for page in corpus}  # Initialize PageRank dictionary

    transition_models = {page: transition_model(corpus, page, DAMPING) for page in pageRanks}

    current_page = random.choice(list(pageRanks.keys()))
    pageRanks[current_page] += 1
    for _ in range(n):
        states = list(transition_models[current_page].keys())
        state_probabilities = list(transition_models[current_page].values())
        current_page = random.choices(states, state_probabilities, k=1)[0] 
        pageRanks[current_page] += 1

        print(f"Page selected {current_page}")

    for page in pageRanks:
        print(f"Number of occurences for page: {page}, {pageRanks[page]}")
        pageRanks[page] /= n
        print(f"Probability for page: {page}, {pageRanks[page]}")

    print("\n\n\n\nStarting Iterative Approach")
    return pageRanks


def iterate_pagerank(corpus, damping_factor):
    N = len(corpus)
    pageRanks = {page: 1 / N for page in corpus}
    convergence_threshold = 0.001

    # Continue the loop until convergence
    while True:
        new_pageRanks = {}
        change = False
        
        for page in corpus:
            # Calculate sum part of the PageRank formula
            sum_part = 0
            for other_page in corpus:
                if page in corpus[other_page]:
                    sum_part += pageRanks[other_page] / len(corpus[other_page])
                elif not corpus[other_page]:  # Handle the case where pages have no outgoing links
                    sum_part += pageRanks[other_page] / N

            new_rank = (1 - damping_factor) / N + damping_factor * sum_part
            new_pageRanks[page] = new_rank

            # Check if the rank change is within the threshold for convergence
            if not change and abs(new_pageRanks[page] - pageRanks[page]) > convergence_threshold:
                change = True

        # Update ranks if any changes occurred beyond the threshold
        if not change:
            break
        
        pageRanks = new_pageRanks

    return pageRanks




if __name__ == "__main__":
    main()
