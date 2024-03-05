package pass_leecode;

/**
 * @author xwp
 * @date 2024/2/23
 * @Description
 */


class ListNode {
    int val;
    ListNode next;
    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

public class reverseGroupK {

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0, head), p0 = dummy;
        p0.next = head;
        ListNode num = head;
        int len = 0;
        while (num != null) {
            len++;
            num = num.next;
        }
        int loop = len / k;
        ListNode preCur = p0.next;
        ListNode preNode = p0;
        for (int i = 0; i < loop; i++) {
            ListNode cur = preCur, pre = null;
            for (int j = 0; j < k; j++) {
                ListNode next = cur.next;
                cur.next = pre;
                pre = cur;
                cur = next;
            }
            preNode.next = pre;
            preNode = preCur;
            preCur = cur;

        }
        preNode.next = preCur;
        return dummy.next;

    }


}

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class reverseGroupK2 {

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode p = head;
        for(int i = 0; i < k; i++){
            if(p == null) return head;
            p = p.next;
        }
        ListNode last = head;
        ListNode[] res = reverse(head,k);
        last.next = reverseKGroup(res[1],k);
        return res[0];

    }
    public ListNode[] reverse(ListNode head, int k){
        ListNode pre = null, now = head;
        while(k != 0){
            ListNode t = now.next;
            now.next = pre;
            pre = now;
            now = t;
            k--;
        }
        ListNode[] res = new ListNode[2];
        res[0] = pre;
        res[1] = now;
        return res;
    }

}
