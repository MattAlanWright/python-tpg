# python-tpg

This is a Python implementation of one of the of the most fantabulous reinforcement learning algorithms to ever grace the technosphere: Tangled Program Graphs, aka TPG. This algorithm is the result of the work done by Dr. Malcolm Heywood and his many students at Dalhousie University. It is a significant extension of linear genetic programming, a good intro to which is found in this paper: https://pdfs.semanticscholar.org/7cb0/f6755e325494afbda4f822026e8e6953ffe1.pdf. If you'd like to go deeper into linear genetic programming itself, the authors of the paper have also written a book simply called *Linear Genetic Programming*. 



## Brief History

In (very very) brief, the road from linear genetic programming to TPG is as follows.

Dr. Heywood and his student Peter Lichodzijewski began developing more sophisticated algorithms on top of canonical linear GP that allowed teams of genetic programs to collaborate when faced with a new exemplar to classify or environment to navigate. This resulted in BidGP, an algorithm in which an entire population of genetic programs comprise the final solution and bid to have their action performed. The paper describing this can be found here: https://web.cs.dal.ca/~mheywood/X-files/Publications/peter-GPEM08.pdf. The result of Bid GP is computationally heavy, but it generalizes very well, tends to improve performance over standard linear GP, and proves the usefulness of evolving collections of GPs to work together on a task.

Malcolm and Peter continued from there to develop Symbiotic Bid-Based GP, or SBB, an algorithm for evolving small teams of genetic programs that leverages the collaborative nature of Bid GP with significantly less computational overhead and redundancy in the final solution. Peter's thesis describing this algorithm can be found here: https://web.cs.dal.ca/~mheywood/Thesis/PLichodzijewski.pdf.

Next, Stephen Kelly came along and, with Dr. Heywood, extended SBB from a flat hierarchy of teams to a graph structure that allows teams of programs to discover ways to leverage each others' behaviour. This algorithm has been used to play Atari like you wouldn't believe. Stephen's thesis describing this can be found here: https://web.cs.dal.ca/~mheywood/Thesis/Kelly-Stephen-PhD-CSCI-June-2018.pdf.

Since then, Malcolm's current student Robert Smith has re-written TPG in every language imaginable, including a particularly pretty Java implementation that has become the de facto standard for implementing TPG. This implementation can be found here: https://github.com/LivinBreezy/TPG-J. In addition to the technical overhaul, Robert has worked with Malcolm to give TPG agents memory, allowing them to perform better in partially observable environments such as VizDoom. Some of his work on VizDoom can be found here: https://web.cs.dal.ca/~mheywood/OpenAccess/open-smith18.pdf.

The implementation found in this repo is heavily based on another Python implementation by one of Dr. Heywood's other students, Ryan Amaral, found here: https://github.com/Ryan-Amaral/PyTPG. If you read his and mine and find that mine looks suspiciously like his, it's because I read his, liked it, and stole liberally.

The descriptions above are embarrassingly brief and don't nearly convey the coolness of those algorithms, so I recommend digging deeper if you have some time and a hankering to read a few academic papers. If I've left anyone out, or ignored work that anyone mentioned above thinks would be a better representation of their contribution, I apologize and am happy to take suggestions on changes to make.
