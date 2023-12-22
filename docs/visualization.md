How to visualize the results
---

We propose using [CosmoGraph](https://cosmograph.app/run/) website as a way to visualize the results.

You will need to use 2 csv files:

- Records file: An adjacency list of source and target node ids. The csv file should have two columns: source,target.
- Metadata file: A list of node ids with its metadata. The csv file should have two columns: id,color. Here color can be the community_id or the hex code of the color that you want.

Once you've entered these two file, set the node appearance color to be `metadata | color`.

Hit start, and the app will visualize the graph for you.

Example configuration so that visualization are consistent:

- Node appearance:
    - node scale: 0.5

- Simulation
    - gravity: 0.5
    - repulsion: 0.5
    - repulsion theta: 0.5
    - link strength: 0.5
    - minimum link distance: 15
    - friction: 0.5
