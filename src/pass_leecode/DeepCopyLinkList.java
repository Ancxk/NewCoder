package pass_leecode;

/**
 *
 * 剑指offer 复制复杂链表
 * @author xwp
 * @date 2023/6/28
 * @Description
 */
public class DeepCopyLinkList {
    public static void main(String[] args) {


    }
    public Node root(Node node){
        Node cur = node;
        while(cur != null){
            Node copy = new Node(cur.val);
            copy.next = cur.next;
            cur.next = copy;
            cur = copy.next;
        }
        cur = node;

        while(cur != null){
            Node random = cur.random;
            Node copy2 = cur.next;
            copy2.random = random.next;
            cur = copy2.next;
        }
        Node t = node;
        Node r = node.next;
        while(r.next != null){
            Node temp = r.next;
            r.next = r.next.next;
            r = r.next.next;
            t.next = temp;
            t = t.next;
        }
        t.next = null;
        return node.next;

    }



}

class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
