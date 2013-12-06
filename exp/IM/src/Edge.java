/**
 * Created with IntelliJ IDEA.
 * User: npbool
 * Date: 12/6/13
 * Time: 7:01 PM
 * To change this template use File | Settings | File Templates.
 */

public class Edge {
    public int node;
    public double activeProb;
    public double negLogProb;
    public Edge(int node, double activeProb){
        this.node = node;
        this.activeProb = activeProb;
        negLogProb = -Math.log(activeProb);
    }
}
