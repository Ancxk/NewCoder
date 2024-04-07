package lc;

/**
 * @author xwp
 * @date 2024/4/7
 * @Description
 */
public class H100 {
    public static void main(String[] args) {


    }
}


/**
 https://leetcode.cn/problems/remove-nth-node-from-end-of-list/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution19 {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0,head),h = head,low = dummy;
        for(int i = 0; i < n; i++){
            h = h.next;
        }
        while(h != null){
            h =h.next;
            low = low.next;
        }
        low.next = low.next.next;
        return dummy.next;
    }
}

/**
 https://leetcode.cn/problems/add-two-numbers/?envType=study-plan-v2&envId=top-100-liked
 * }
 */
class Solution2 {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0),h = dummy;
        int k = 0;
        while(l1 != null || l2 != null || k != 0){
            int lnum = l1 == null ? 0 : l1.val;
            int rnum = l2 == null ? 0 : l2.val;
            int val = lnum+rnum+k;
            k =  val / 10;
            ListNode p = new ListNode(val%10);
            h.next = p;
            h = p;
            if(l1 != null) l1 = l1.next;
            if(l2 != null) l2 = l2.next;
        }
        return dummy.next;
    }
}

/*
https://leetcode.cn/problems/swap-nodes-in-pairs/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution24 {
    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode hnext = head.next.next;
        ListNode p = head.next;
        p.next = head;
        head.next = swapPairs(hnext);
        return p;
    }
}


/**
 * https://leetcode.cn/problems/reverse-nodes-in-k-group/
 */
class Solution25 {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode p = head;
        for(int i =0; i < k; i++){
            if(p == null){
                return head;
            }
            p = p.next;
        }
        ListNode pre = null,h = head;
        for(int i = 0;i < k; i++){
            ListNode q = h.next;
            h.next = pre;
            pre = h;
            h = q;
        }
        head.next = reverseKGroup(p,k);
        return pre;
    }

}
