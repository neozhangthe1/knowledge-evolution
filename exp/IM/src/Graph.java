import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Created with IntelliJ IDEA.
 * User: npbool
 * Date: 12/6/13
 * Time: 7:00 PM
 * To change this template use File | Settings | File Templates.
 */
class TreeEdge {
    int node;
    TreeEdge next;
    double ppToDst;

    public TreeEdge(int node, TreeEdge next, double ppToDst){
        this.node = node;
        this.next = next;
        this.ppToDst = ppToDst;
    }
}
public class Graph {
    public Edge[][] graph;
    public Edge[][] revGraph;
    public HashMap<Integer, Double> pp[];

    public int[] inDeg;
    public int[] outDeg;

    public int node_count = 0;
    public boolean loadFile(String filename){
        try{
            BufferedReader bf = new BufferedReader(new FileReader(filename));
            node_count = Integer.parseInt(bf.readLine());
            graph = new Edge[node_count][];
            inDeg = new int[node_count];
            outDeg = new int[node_count];
            pp = new HashMap[node_count];

            for(int i=0;i<node_count;++i){
                String[] line = bf.readLine().split("\t");
                int neighbor_count = Integer.parseInt(line[0]);
                outDeg[i] = neighbor_count;
                graph[i] = new Edge[neighbor_count];
                pp[i] = new HashMap<Integer, Double>();
                for(int j=0;j<neighbor_count;++j){
                    int neighbor = Integer.parseInt(line[j*2+1]);
                    inDeg[neighbor]++;
                    double active_prob = Double.parseDouble(line[j*2+2]);
                    graph[i][j] = new Edge(neighbor, active_prob);
                    pp[i].put(neighbor, active_prob);
                }
            }

            revGraph = new Edge[node_count][];
            int[] pos = new int[node_count];
            for(int i=0;i<node_count;++i){
                revGraph[i] = new Edge[inDeg[i]];
            }
            for(int i=0;i<node_count;++i){
                for(Edge neighborEdge : graph[i]){
                    int end = neighborEdge.node;
                    revGraph[end][pos[end]] = new Edge(i, neighborEdge.activeProb);
                    ++pos[end];
                }
            }
            bf.close();
            return true;
        } catch (Exception e){
            e.printStackTrace();
            return false;
        }
    }

    public List<TreeEdge> calculateMIIA(int dst, double theta){
        ArrayList<TreeEdge> miia = new ArrayList<TreeEdge>();
        double negLogTheta = -Math.log(theta);

        double dist[] = new double[node_count];
        TreeEdge next[] = new TreeEdge[node_count];

        for(int i=0;i<node_count;++i){
            dist[i] = 10000;
        }
        dist[dst] = 0;
        boolean chosen[] = new boolean[node_count];
        int remain = node_count;
        while(remain>0){
            double minDist = 1000;
            int selection = 0;
            for(int i=0;i<node_count;++i){
                if(!chosen[i] && minDist>dist[i]){
                    minDist = dist[i];
                    selection = i;
                }
            }
            chosen[selection] = true;
            if(minDist > negLogTheta){
                break;
            }
            TreeEdge selNode = new TreeEdge(selection, next[selection], Math.exp(-minDist));
            //if(selection != dst){
                miia.add(selNode);
            //}
            for(Edge edge : revGraph[selection]){
                if(dist[edge.node] > dist[selection] + edge.negLogProb){
                    dist[edge.node] = dist[selection] + edge.negLogProb;
                    next[edge.node] = selNode;
                }
            }
            --remain;
        }
        return miia;
    }

    double alpha[];
    HashSet<Integer> seed;
    List<TreeEdge>[] miia;
    List<Integer>[] inMIIA;
    double[] ap;
    double incCov[];
    public double calculateAlpha(int v, TreeEdge uEdge){
        int u = uEdge.node;
        if(v==u){
            alpha[v] = 1.0;
            return 1.0;
        }

        TreeEdge wEdge = uEdge.next;
        int w = wEdge.node;
        if(seed.contains(w)){
            alpha[u] = 0.0;
            return 0;
        }
        double alpha_v_u = 1;
        double alpha_v_w = 0;

        alpha_v_w = calculateAlpha(v, wEdge);
        alpha_v_u = alpha_v_w * pp[u].get(w);
        for(Edge uin : revGraph[w]){
            if(uin.node != u){
                alpha_v_u *= (1-ap[uin.node]*pp[uin.node].get(w));
            }
        }
        alpha[u] = alpha_v_u;
        return alpha_v_u;
    }
    public double calculateAP(int u){
        if(seed.contains(u)){
            ap[u] = 1;
        } else if(outDeg[u] == 0){
            ap[u] = 0;
        } else {
            ap[u] = 1;
            for(Edge wEdge : revGraph[u]){
                int w = wEdge.node;
                ap[u] *= 1-ap[w]*pp[w].get(u);
            }
            ap[u] = 1 - ap[u];
        }
        return ap[u];
    }
    public Set<Integer> influenceMaxization(int K, double theta){
        Log.print("INIT");
        seed = new HashSet<Integer>();
        miia = new List[node_count];
        inMIIA = new List[node_count];

        incCov = new double[node_count];
        ap = new double[node_count];
        alpha = new double[node_count];
        for(int i=0;i<node_count;++i){
            inMIIA[i] = new ArrayList<Integer>();
        }

        for(int v=0;v<node_count;++v){
            List<TreeEdge> miia_v = calculateMIIA(v, theta);
            miia[v] = miia_v;
            for(TreeEdge uEdge : miia_v){
                inMIIA[uEdge.node].add(v);
            }
            for(TreeEdge uEdge : miia_v){
                calculateAlpha(v, uEdge);
            }
            for(TreeEdge uEdge : miia_v){
                incCov[uEdge.node] += alpha[uEdge.node]*(1-ap[uEdge.node]);
            }
        }
        //main loop
        Log.print("Main Loop");
        boolean chosen[] = new boolean[node_count];
        for(int i=0;i<K;++i){
            Log.print("Loop "+i);
            int u = -1;
            double maxIncCov = -1;
            for(int s=0;s<node_count;++s){
                if(!chosen[s]&& incCov[s]> maxIncCov){
                    u = s;
                    maxIncCov = incCov[s];
                }
            }

            for(int v : inMIIA[u]){
                if(seed.contains(v)) continue;
                for(TreeEdge wEdge : miia[v]){
                    int w = wEdge.node;
                    if(seed.contains(w)) continue;

                    incCov[w] -= alpha[w] * (1-ap[w]);
                }
            }
            seed.add(u);
            chosen[u] = true;

            for(TreeEdge vEdge : miia[u]){
                int v = vEdge.node;
                if(seed.contains(v)) continue;

                for(TreeEdge wEdge : miia[v]){
                    calculateAP(wEdge.node);
                }
                for(TreeEdge wEdge : miia[v]){
                    calculateAlpha(v, wEdge);
                }

                for(TreeEdge wEdge : miia[v]){
                    int w = wEdge.node;
                    if(seed.contains(w)) continue;

                    incCov[w] += alpha[w] * (1-ap[w]);
                }
            }
        }
        return seed;
    }
    public Graph(){}
}
