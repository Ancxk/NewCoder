package pass_leecode;

import java.util.HashMap;
import java.util.Map;

/**
 * @author xwp
 * @date 2023/8/12
 * @Description
 */
public class LRU {
    public static void main(String[] args) {

    }
}


class LRUCache {
    //节点
    static class Node{
        int key;
        int value;
        Node pre;
        Node next;

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    Node dummyHead,dummyTail;
    int len = 0;

    Map<Integer,Integer> map = new HashMap<>();

    public LRUCache(int capacity) {
        len = capacity;
        dummyHead = new Node(-1, -1);
        dummyTail = new Node(-1, -1);
        dummyHead.next = dummyTail;
    }

    public int get(int key) {
        if(!map.containsKey(key)) return -1;
        int value = map.get(key);
        Node p = dummyHead.next;
        while(p.key != key){
            p = p.next;
        }
        p.pre.next = p.next;
        p.next.pre = p.pre;
        p.pre = dummyHead;
        p.next = dummyHead.next;
        dummyHead.next = p;

        return value;
    }

    public void put(int key, int value) {
        if(map.containsKey(key)){
            Node p = dummyHead.next;
            while(p.key != key){
                p = p.next;
            }
            p.value = value;
            map.put(key,value);
            get(key);
        }else{
            if(map.size() == len){
                Node d = dummyTail.pre;
                d.pre.next = dummyTail;
                dummyTail.pre = d.pre;
                d.pre = null;
                d.next = null;

                map.remove(d.key);
                len--;
            }
            Node node = new Node(key, value);
            node.next = dummyHead.next;
            dummyHead.next.pre = node;

            node.pre = dummyHead;
            dummyHead.next = node;
            len++;
            map.put(key,value);

        }

    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
