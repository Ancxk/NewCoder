package lc;

import java.util.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

/**
 * @author xwp
 * @date 2024/4/7
 * @Description
 */



/**
 * https://leetcode.cn/problems/remove-nth-node-from-end-of-list/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution19 {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head), h = head, low = dummy;
        for (int i = 0; i < n; i++) {
            h = h.next;
        }
        while (h != null) {
            h = h.next;
            low = low.next;
        }
        low.next = low.next.next;
        return dummy.next;
    }

    public ListNode removeNthFromEnd2(ListNode head, int n) {
        ListNode dummy = new ListNode(),h = dummy;
        dummy.next = head;
        for (int i = 0; i < n-1; i++) {
            head = head.next;
        }
        while (head.next != null){
            head = head.next;
            h = h.next;
        }
        h.next = head;
        return dummy.next;
    }
}

/**
 * https://leetcode.cn/problems/add-two-numbers/?envType=study-plan-v2&envId=top-100-liked
 * }
 */
class Solution2 {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0), h = dummy;
        int k = 0;
        while (l1 != null || l2 != null || k != 0) {
            int lnum = l1 == null ? 0 : l1.val;
            int rnum = l2 == null ? 0 : l2.val;
            int val = lnum + rnum + k;
            k = val / 10;
            ListNode p = new ListNode(val % 10);
            h.next = p;
            h = p;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        return dummy.next;
    }
}

/*
https://leetcode.cn/problems/swap-nodes-in-pairs/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution24 {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
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
        for (int i = 0; i < k; i++) {
            if (p == null) {
                return head;
            }
            p = p.next;
        }
        ListNode pre = null, h = head;
        for (int i = 0; i < k; i++) {
            ListNode q = h.next;
            h.next = pre;
            pre = h;
            h = q;
        }
        head.next = reverseKGroup(p, k);
        return pre;
    }

}

/*
https://leetcode.cn/problems/merge-k-sorted-lists/?envType=study-plan-v2&envId=top-100-liked
 */

/**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode() {}
 * ListNode(int val) { this.val = val; }
 * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution23 {
    public ListNode mergeKLists(ListNode[] lists) {
        return merge3(lists);
    }

    public ListNode merge1(ListNode[] lists) {
        if (lists.length == 0) return null;
        if (lists.length == 1) return lists[0];
        ListNode p = merge(lists[0], lists[1]);
        for (int i = 2; i < lists.length; i++) {
            p = merge(p, lists[i]);
        }
        return p;
    }

    public ListNode merge2(ListNode[] list, int l, int r) {
        if (r < l) return null;
        if (r == l) return list[r];
        int mid = (l + r) / 2;
        ListNode a = merge2(list, l, mid), b = merge2(list, mid + 1, r);
        return merge(a, b);
    }

    public ListNode merge(ListNode q, ListNode p) {
        ListNode dummy = new ListNode(), head = dummy;
        ListNode a = q, b = p;
        while (a != null && b != null) {
            if (a.val >= b.val) {
                dummy.next = b;
                b = b.next;
            } else {
                dummy.next = a;
                a = a.next;
            }
            dummy = dummy.next;
        }
        if (a != null) dummy.next = a;
        if (b != null) dummy.next = b;
        return head.next;
    }

    public ListNode merge3(ListNode[] lists) {
        PriorityQueue<ListNode> que = new PriorityQueue<>(Comparator.comparingInt(l -> l.val));
        for (ListNode node : lists) {
            if (node != null) que.offer(node);
        }
        ListNode dummy = new ListNode(), cur = dummy;
        while (!que.isEmpty()) {
            ListNode q = que.poll();
            cur.next = q;
            cur = q;
            if (q.next != null) {
                que.offer(q.next);
            }
        }
        return dummy.next;
    }
}


/*
https://leetcode.cn/problems/copy-list-with-random-pointer/?envType=study-plan-v2&envId=top-100-liked
 */

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

class Solution138 {
    public Node copyRandomList(Node head) {
        if (head == null) return head;
        Node now = head;
        while (now != null) {
            Node copy = new Node(now.val);
            Node next = now.next;
            now.next = copy;
            copy.next = next;
            now = next;
        }
        now = head;
        while (now != null) {
            Node nr = now.random;
            if (nr != null) {
                now.next.random = nr.next;
            } else {
                now.next.random = nr;
            }
            now = now.next.next;
        }
        now = head;
        Node res = head.next;
        while (now != null) {
            Node copy = now.next, next = copy.next;
            if (copy.next != null) copy.next = copy.next.next;
            now.next = next;
            now = now.next;
        }
        return res;
    }
}

/*
https://leetcode.cn/problems/lru-cache/description/?envType=study-plan-v2&envId=top-100-liked
 */

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */

/*
超时了wc,没必要遍历阿wc。直接删除，再添加...
 */
class LRUCache {
    static class Node {
        int k, v;
        Node pre;
        Node next;

        public Node(int k, int v) {
            this.k = k;
            this.v = v;
        }
    }

    int c;
    Map<Integer, Integer> map = new HashMap<>();
    Node dummy = new Node(-1, 0);
    Node dummyL = new Node(-1, 0);

    public LRUCache(int capacity) {
        c = capacity;
        dummyL.pre = dummy;
        dummy.next = dummyL;
    }

    public int get(int key) {
        Integer res = map.get(key);
        if (res == null) return -1;
        Node t = dummy.next;
        while (t != null) {
            if (t.k == key) {
                t.pre.next = t.next;
                t.next.pre = t.pre;
                break;
            }
            t = t.next;
        }
        Node temp = dummy.next;
        dummy.next = t;
        t.pre = dummy;
        t.next = temp;
        temp.pre = t;
        return res;
    }

    public void put(int key, int value) {
        Integer res = map.get(key);
        if (res == null) {
            map.put(key, value);
            Node t = dummy.next;
            Node node = new Node(key, value);
            dummy.next = node;
            node.pre = dummy;
            node.next = t;
            t.pre = node;
            if (map.size() > c) {
                Node del = dummyL.pre;
                del.pre.next = del.next;
                del.next.pre = del.pre;
                int rem = del.k;
                map.remove(rem);
            }
        } else {
            this.get(key);
            map.put(key, value);
        }
    }
}


class LRUCache2 {
    static class Node {
        int k, v;
        Node pre;
        Node next;

        public Node(int k, int v) {
            this.k = k;
            this.v = v;
        }
    }

    int c;
    Map<Integer, Node> map = new HashMap<>();
    Node dummy = new Node(-1, 0);
    Node dummyL = new Node(-1, 0);

    public LRUCache2(int capacity) {
        c = capacity;
        dummyL.pre = dummy;
        dummy.next = dummyL;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) return -1;
        unLink(node);
        addHead(node);
        return node.v;
    }


    public void unLink(Node node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
    }

    public void addHead(Node node) {
        Node tem = dummy.next;
        dummy.next = node;
        node.pre = dummy;
        node.next = tem;
        tem.pre = node;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.v = value;
            unLink(node);
            addHead(node);
        } else {
            Node t = new Node(key, value);
            addHead(t);
            map.put(key, t);
        }
        if (map.size() > c) {
            Node del = dummyL.pre;
            unLink(del);
            map.remove(del.k);
        }
    }
}


/*
https://leetcode.cn/problems/binary-tree-inorder-traversal/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution94 {
    List<Integer> res = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        dfs(root);
        return res;
    }

    public void dfs(TreeNode node) {
        if (node == null) return;
        dfs(node.left);
        res.add(node.val);
        dfs(node.right);
    }
}


class Solution104 {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}

class Solution226 {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode n = root.left;
        root.left = root.right;
        root.right = n;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }
}


/*
轴。。。
 */
class Solution543 {
    int res = 0;
    Map<TreeNode,Integer> m = new HashMap<>();
    public int diameterOfBinaryTree(TreeNode root) {
        if (root ==  null) return 0;
        int l = diameterOfBinaryTree(root.left);
        int r = diameterOfBinaryTree(root.right);
        res = Math.max(res, l + r + 1);
        return res;
    }

}

/*
https://leetcode.cn/problems/binary-tree-level-order-traversal/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution102 {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
            Deque<TreeNode> que = new LinkedList<>();
            if (root == null) return res    ;
            que.add(root);
            while (!que.isEmpty()){
                int i = que.size();
                List<Integer> ans = new ArrayList<>();
                for (int j = 0; j < i; j++) {
                    TreeNode pop = que.pop();
                    ans.add(pop.val);
                    if (pop.left != null) que.add(pop.left);
                    if (pop.right != null) que.add(pop.right);
                }
                res.add(ans);
            }
            return res  ;
    }
}

/*
https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked
 */

class Solution108 {
    public TreeNode sortedArrayToBST(int[] nums) {
        return dfs(nums,0,nums.length-1);
    }
    public TreeNode dfs(int[] nums,int i, int j){
        if(i>j) return null;
        if(i == j) return new TreeNode(nums[i]);
        int mid = (i+j)/2;
        int num = nums[mid];
        TreeNode treeNode = new TreeNode(num);
        treeNode.left = dfs(nums,i,mid-1);
        treeNode.right = dfs(nums,mid+1,j);
        return treeNode;
    }
}


/*
https://leetcode.cn/problems/validate-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked
 */
 class Solution98 {
     TreeNode pre = null;
    public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        boolean l = isValidBST(root.left);
        boolean cur = true;
        if(pre != null){
            cur = pre.val > root.val;
        }
        pre = root;
        boolean r = isValidBST(root.right);
        return cur && r && l;
    }
}


/*
https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=study-plan-v2&envId=top-100-liked
 */

class Solution230 {
    int k,t;
    public int kthSmallest(TreeNode root, int k) {
        this.k = k;
        return dfs(root);
    }
    public int dfs(TreeNode root){
        if (root == null) return -1;
        int j = dfs(root.left);
        if(j != -1) return j;
        t++;
        if(t == k) return root.val;
        return dfs(root.right);
    }
}

/*
https://leetcode.cn/problems/binary-tree-right-side-view/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution199 {
    List<Integer> r = new ArrayList<>();
    public List<Integer> rightSideView2(TreeNode root) {
        int[] ans = new int[100];
        Arrays.fill(ans,-101);
        dfs(root,0,ans);
        for (int an : ans) {
            if (an == -101) break;
            r.add(an);
        }
        return r;
    }
    public void dfs(TreeNode node,int k,int[] deep){
        if(node == null) return;
        if(deep[k] == -101) deep[k] = node.val;
        dfs(node.right,k+1,deep);
        dfs(node.left,k+1,deep);
    }
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        Deque<TreeNode> que = new LinkedList<>();
        que.add(root);
        if(root == null) return ans;
        while(!que.isEmpty()){
            int n = que.size();
            for(int i = 0; i < n; i++){
                TreeNode node =  que.pop();
                if(i == 0){
                    ans.add(node.val);
                }
                if(node.right != null) que.add(node.right);
                if(node.left != null) que.add(node.left);
            }
        }
        return ans;
    }
}
/*
https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution114 {
    //前序遍历
    TreeNode pre = null;
    public void flatten2(TreeNode root) {
        if(root == null) return;
        if(pre != null){
            pre.right = root;
            pre.left = null;
        }
        pre = root;
        TreeNode ri = root.right;
        flatten2(root.left);
        flatten2(ri);
    }

    //后序遍历
    public void flatten3(TreeNode root) {
        if(root == null) return;
        flatten3(root.right);
        flatten3(root.left);
        root.right = pre;
        root.left = null;
        pre = root;
    }

    //顺序数组
    public void flatten(TreeNode root) {
        if(root == null) return;
        List<TreeNode> de = new ArrayList<>();
        dfs(root,de);
        TreeNode head = de.get(0);
        head.left = null;
        for (int i = 1; i < de.size(); i++) {
            TreeNode treeNode = de.get(i);
            head.right = treeNode;
            head = treeNode;
            head.left = null;
        }
    }
    public void dfs(TreeNode n,List<TreeNode> de){
        if(n == null) return;
        de.add(n);
        dfs(n.left,de);
        dfs(n.right,de);
    }
}


/*
https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution105 {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return build(preorder,inorder);
    }
    public TreeNode build(int[] pre,int[] in){
        if (pre.length == 0) return null;
        int val = pre[0];
        int index = getIndex(in,val);
        int[] lpre = Arrays.copyOfRange(pre, 1, 1+index);
        int[] rpre = Arrays.copyOfRange(pre, index+1, pre.length);
        int[] lin = Arrays.copyOfRange(in, 0, index);
        int[] rin = Arrays.copyOfRange(pre, index+1, in.length);
        TreeNode treeNode = new TreeNode(val);
        treeNode.left = build(lpre  ,lin);
        treeNode.right = build(rpre  ,rin);
        return treeNode;
    }
    public int getIndex(int[] pre,int val){
        for (int i = 0; i < pre.length; i++) {
            if(pre[i] == val) return i;
        }
        return 0;
    }
}

/*
https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
 */

class Solution106 {
    public TreeNode buildTree(int[] in, int[] post) {
        if(in.length == 0) return null;
        int len = post.length;
        int val = post[len - 1];
        int index = getIndex(in, val);
        TreeNode treeNode = new TreeNode(val);
        treeNode.left= buildTree(Arrays.copyOfRange(in,0,index),Arrays.copyOfRange(post,0,index));
        treeNode.right = buildTree(Arrays.copyOfRange(in,index+1,len),Arrays.copyOfRange(post,index,len-1));
        return treeNode;
    }
    public int getIndex(int[] pre,int val){
        for (int i = 0; i < pre.length; i++) {
            if(pre[i] == val) return i;
        }
        return 0;
    }
}

class Solution889 {
    public TreeNode constructFromPrePost(int[] in, int[] post) {
        if(in.length == 0) return null;
        int len = in.length;
        int val = in[0];
        TreeNode treeNode = new TreeNode(val);
        if (in.length > 1){
            int i = in[1];
            int index = getIndex(post, i);
            treeNode.left= constructFromPrePost(Arrays.copyOfRange(in,1,index+2),Arrays.copyOfRange(post,0,index+1));
            treeNode.right = constructFromPrePost(Arrays.copyOfRange(in,index+2,len),Arrays.copyOfRange(post,index+1,len-1));
        }
        return treeNode;
    }
    public int getIndex(int[] pre,int val){
        for (int i = 0; i < pre.length; i++) {
            if(pre[i] == val) return i;
        }
        return 0;
    }
}

/*
https://leetcode.cn/problems/path-sum-iii/description/?envType=study-plan-v2&envId=top-100-liked
 */

class Solution437 {
    int res;
    public int pathSum(TreeNode root, int targetSum) {
        if(root == null) return 0;
        dfs(root,0,targetSum);
        pathSum(root.left,targetSum);
        pathSum(root.right,targetSum);
        return res;
    }
    public void dfs(TreeNode node, int sum,int tar){
        if(node == null) return;
        sum += node.val;
        sum %= 1000000007;
        if(sum == tar) res++;
        dfs(node.left,sum,tar);
        dfs(node.right,sum,tar);
    }

    //前缀和
    Map<Long,Integer> m = new HashMap<>();
    int ans;
    public int pathSum2(TreeNode root, int targetSum) {
        m.put(0L,1);
        dfs2(root,0,targetSum);
        return ans;
    }
    public void dfs2(TreeNode node,long sum,int tar){
        if(node == null) return;
        sum += node.val;
        ans += m.getOrDefault(sum - tar,0);
        //m.put(sum,m.getOrDefault(sum,0)+1);
        m.merge(sum, 1, Integer::sum);
        dfs2(node.left,sum,tar);
        dfs2(node.right,sum,tar);
        m.merge(sum, -1, Integer::sum);
    }
}
/*
https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/?envType=study-plan-v2&envId=top-100-liked
 */

class Solution236 {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null) return null;
        if(root == p || root == q) return root;
        TreeNode l = lowestCommonAncestor(root.left,p,q);
        TreeNode r = lowestCommonAncestor(root.right,p,q);
        if(l != null && r != null) return root;
        if (l != null) return l;
        if(r != null) return r;
        return null;
    }
}

/*
https://leetcode.cn/problems/binary-tree-maximum-path-sum/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution124 {
    int ans = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return ans;
    }
    public int dfs(TreeNode node){
        if(node == null) return 0;
        int l = dfs(node.left);
        int r = dfs(node.right);
        int re = node.val;
        int ns = Math.max(l,r);
        if(ns > 0) re+=ns;
        int t = node.val + (Math.max(l, 0)) +(Math.max(r, 0));
        ans = Math.max(ans,t);
        return re;
    }
}


/*
https://leetcode.cn/problems/number-of-islands/?envType=study-plan-v2&envId=top-100-liked

 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。
示例 1：
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
 */
class Solution200 {
    int n,m,ans;
    public int numIslands(char[][] grid) {
        Deque<int[]> que = new LinkedList<>();
        n = grid.length;
        m= grid[0].length;
        for(int i = 0; i < n;i++){
            for(int j = 0; j < m; j++){
                if(grid[i][j] == '1')  ans++;
                dfs(grid,i,j);
            }
        }
        return ans;

    }
    public void dfs(char[][] grid,int i,int j){
        if(i < 0 || i > n-1 || j < 0 || j > m-1 || grid[i][j] != '1') return;
        grid[i][j]++;
        dfs(grid,i+1,j);
        dfs(grid,i-1,j);
        dfs(grid,i,j+1);
        dfs(grid,i,j-1);
    }
}

/*
https://leetcode.cn/problems/rotting-oranges/?envType=study-plan-v2&envId=top-100-liked

 */
class Solution994 {
    int ans = -1, n, m;
    int[][] v;

    public int orangesRotting(int[][] g) {
        n = g.length;
        m = g[0].length;
        v = new int[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (g[i][j] == 2) {
                    dfs(g, i, j, i, j);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (v[i][j] == 0 && g[i][j] == 1)
                    return -1;
                ans = Math.max(ans, v[i][j]);
            }
        }
        return ans;

    }

    public void dfs(int[][] grid, int i, int j, int oi, int oj) {
        if (i < 0 || i > n - 1 || j < 0 || j > m - 1 || grid[i][j] == 0)
            return;
        if(grid[i][j] == 2 &&(oi != i || oj != j) ) return;
        int old = v[i][j];
        int last = v[oi][oj];
        if (grid[i][j] == 1) {
            if (v[i][j] == 0)
                v[i][j] = last+1;
            else if (v[i][j] < last + 1){
                return;
            }
            v[i][j] = Math.min(v[i][j], last + 1);
        }
        dfs(grid, i + 1, j, i, j);
        dfs(grid, i - 1, j, i, j);
        dfs(grid, i, j + 1, i, j);
        dfs(grid, i, j - 1, i, j);
    }


    public int bfs(int[][] g){
        int[][] move = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
        n = g.length;
        m = g[0].length;
        Deque<int[]> que = new LinkedList<>();
        int ct = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if(g[i][j] == 2) que.add(new int[]{i,j,0});
                if(g[i][j] == 1 || g[i][j] == 2) ct++;
            }
        }
        if(ct == 0) return 0;
        int r = 0;
        while (!que.isEmpty()){
            int len = que.size();
            ct -= len;
            r++;
            for (int i = 0; i < len; i++) {
                int[] index = que.pop();
                int a = index[0], b = index[1],c = index[2];
                for (int i1 = 0; i1 < 4; i1++) {
                    int x = a+move[i1][0], y = b+move[i1][1];
                    if(x < 0 || x > n-1 || y < 0 || y > m-1 || g[x][y] != 1){
                        continue;
                    }
                    g[x][y] = 2;
                    que.add(new int[]{x,y,c+1});
                }
            }
        }
        return ct == 0 ?  r : -1 ;
    }
}

/*
https://leetcode.cn/problems/course-schedule/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution207 {

    public boolean canFinish(int numCourses, int[][] prerequisites) {


        return false;
    }
    public boolean canFinish2(int numCourses, int[][] prerequisites) {
        //int n = prerequisites.length, m = prerequisites[0].length;
        List<Integer>[] g = new ArrayList[numCourses+1];
        Arrays.setAll(g,i -> new ArrayList<Integer>());
        int[] in = new int[numCourses];
        for(int[] h : prerequisites){
            g[h[1]].add(h[0]);
            in[h[0]]++;
        }
        Deque<Integer> que = new LinkedList<>();
        for(int i = 0; i < numCourses; i++){
            if(in[i] == 0) que.add(i);
        }
        int res = 0;
        while(!que.isEmpty()){
            int t = que.poll();
            res++;
            for(int k : g[t]){
                in[k]--;
                if(in[k] == 0) que.add(k);
            }
        }
        return res == numCourses;

    }
}

/*
https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-100-liked
 */
/**
 * 字典树。。
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
class Trie {
    boolean isEnd;
    Trie[] chids ;

    public Trie() {
        chids = new Trie[26];
    }

    public void insert(String word) {
        Trie cur = this;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            int inx = c-'a';
            if(cur.chids[inx] == null)    {
                cur.chids[inx] = new Trie();
            }
            cur = chids[inx];
        }
        cur.isEnd =true;
    }

    public boolean search(String word) {
        Trie node =searchWith(word);
        return node  != null && node.isEnd;

    }

    public boolean startsWith(String prefix) {
        return searchWith(prefix) != null;
    }

    public Trie searchWith(String prefix) {
        Trie cur = this;
        for (int i = 0; i < prefix.length(); i++) {
            char c = prefix.charAt(i);
            int inx = c-'a';
            if(cur.chids[inx] == null)    {
                return null;
            }
        }
        return cur;
    }
}


/*
https://leetcode.cn/problems/permutations/?envType=study-plan-v2&envId=top-100-liked
 */

class Solution46_2 {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> r = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        dfs(nums,new int[30]);
        return res  ;
    }


    public void dfs2(int[] nums,int i){
        if(i== nums.length) {
            res.add(Arrays.stream(nums).boxed().collect(Collectors.toList()));
            return;
        }
        for (int j = i; j < nums.length; j++) {
            swap(nums,j,i);
            dfs2(nums,j+1);
            swap(nums,j,i);
        }

    }
    public void swap(int[] nums,int i, int j){
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }


    //sb写法
    public void dfs(int[]nums,int[] m){
        if(r.size() == nums.length){
            res.add(new ArrayList<>(r));
            return;
        }
        for (int num : nums) {
            if (m[num + 10] == 1) continue;
            m[num + 10] = 1;
            r.add(num);
            dfs(nums, m);
            r.remove(r.size() - 1);
            m[num + 10] = 0;
        }
    }
}

/*
https://leetcode.cn/problems/3sum/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution15_2 {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> ans = new ArrayList<>();
    public List<List<Integer>> threeSum(int[] nums) {
        ans.sort((o1,o2)->o1-o2);
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if(i > 0 && nums[i] == nums[i-1]) continue;
            int j = i+1,k = nums.length-1;
            while (j<k){
                int n = nums[j]+  nums[k]+nums[i];
                if(n == 0){
                    ans.add(nums[i]);
                    ans.add(nums[j]);
                    ans.add(nums[k]);
                    res.add(new ArrayList<>(ans));
                    ans.clear();
                    j++;
                    k--;
                    while (j < k && nums[j] == nums[j-1]){
                        j++;
                    }
                    while (j < k && nums[k] == nums[k+1]){
                        k--;
                    }
                }else if (n > 0){
                    k--;
                }else{
                    j++;
                }
            }
        }
        return res;
    }
}
/*
https://leetcode.cn/problems/smallest-k-lcci/
 */
class Solution17_14 {
    public int[] smallestK(int[] arr, int k) {
        Arrays.sort(arr);
        return Arrays.copyOfRange(arr,0,k);
    }
    public int[] smallestK2(int[] arr, int k) {
        PriorityQueue<Integer> que = new PriorityQueue<>(4, Comparator.comparingInt(o -> o));
        for (int i : arr) {
            que.add(i);
            if(que.size() > k) que.poll();
        }
        return que.stream().mapToInt(Integer::intValue).toArray();
    }
}

class Solution124_2 {
    int ans = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        get(root);
        return ans;
    }
    public int get(TreeNode root){
        if(root == null) return 0;
        int l = get(root.left);
        int r = get(root.right);
        int val = root.val + (Math.max(l, 0)) + (Math.max(r, 0));
        ans = Math.max(ans,val);
        int n = Math.max(Math.max(l,r),0) ;
        return n + root.val;
    }
}

class Solution437_2 {
    int ans = 0;
    Map<Long,Integer> m = new HashMap<>();
    public int pathSum(TreeNode root, int targetSum) {
        m.put(0L,1);
        path_v2(root,targetSum,0);
        return ans;
    }

    public void dfs(TreeNode root, int tar){
        if(root == null) return ;
        path(root,tar);
        dfs(root.left,tar);
        dfs(root.right,tar);
    }
    public void path(TreeNode root, long tar){
            if(root == null) return;
            long n = tar-root.val;
            if(n == 0) ans++;
            path(root.left,n);
            path(root.right,n);
    }


    public void path_v2(TreeNode r,int tar,long sum){
        if(r == null) return;
        long val = sum+r.val -tar;
        int t = m.getOrDefault(val,0);
        ans+=t;
        m.merge(sum+r.val,1,Integer::sum);
        path_v2(r.left,tar,sum+r.val);
        path_v2(r.right,tar,sum+r.val);
        m.merge(sum+r.val,-1,Integer::sum);
    }
}



class LRUCacheV2 {
    static class Node{
        int val,k;
        Node pre,next;
        public Node(int val,int k){
            this.val = val;
            this.k = k;
        }
    }
    Map<Integer,Node> m = new HashMap<>();
    int c;
    Node dummy;
    Node last;
    public LRUCacheV2(int capacity) {
        c = capacity;
        dummy = new Node(0,0);
        last = new Node(0,0);
        dummy.next=  last;
        last.pre = dummy;
    }

    public int get(int key) {
        Node node = m.get(key);
        if(node == null) return -1;
        first(node);
        return node.val;
    }

    public void first(Node node){
        node.pre.next = node.next;
        node.next.pre = node.pre;
        Node p = dummy.next;
        dummy.next = node;
        node.pre  = dummy;
        node.next = p;
        p.pre  = node;
    }
    public void put(int key, int value) {
        Node node1 = m.get(key);
        if(node1 != null && node1.val == value) return;
        c++;
        Node node = new Node(value,key);
        Node p = dummy.next;
        dummy.next = node;
        node.pre  = dummy;
        node.next = p;
        p.pre  = node;
        m.put(key,node);
        if(m.size() > c){
            pushLast();

        }
    }
    public void pushLast(){
        Node t = last.pre;
        last.pre.pre.next = last;
        last.pre = last.pre.pre;
        m.remove(t.k);
    }
}


/*
https://leetcode.cn/problems/subsets/description/?envType=study-plan-v2&envId=top-100-liked
 */


/*
为啥
 */
class Solution78_2 {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> ans = new ArrayList<>();
    public List<List<Integer>> subsets(int[] nums) {
    /*
        1,2,3,4
        1,2,3,4  12  13 14  23 24
     */
        dfs(nums,0);
        return res;
    }
    public void dfs(int[] nums,int i){
        res.add(new ArrayList<>(ans));
        if(i == nums.length) return;
        for (int j = i; j < nums.length; j++) {
            ans.add(nums[j]);
            dfs(nums,j+1);
            ans.remove(ans.size()-1);
        }
    }
}


/*
https://leetcode.cn/problems/letter-combinations-of-a-phone-number/?envType=study-plan-v2&envId=top-100-liked
 */


class Solution17 {
    List<String> res = new ArrayList<>();
    Map<Character,String> m = new HashMap<>();
    StringBuilder b = new StringBuilder();
    public List<String> letterCombinations(String digits) {
        m.put('2',"abc");
        m.put('3',"def");
        m.put('4',"ghi");
        m.put('5',"jkl");
        m.put('6',"mno");
        m.put('7',"pqrs");
        m.put('8',"tuv");
        m.put('9',"wxyz");
        dfs(digits,0);
        return res;
    }
    public void dfs(String digits,int i){
        if(b.length() == digits.length()) {
            if(b.length() != 0) res.add(b.toString());
            return;
        }
        char c = digits.charAt(i);
        String s = m.get(c);
        for (int j = 0; j < s.length(); j++) {
            b.append(s.charAt(j));
            dfs(digits,i+1);
            b.deleteCharAt(b.length()-1);
        }
    }
}

/*
https://leetcode.cn/problems/combination-sum/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution39 {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> ans = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        dfs(candidates,0,0,target);
        return res;
    }

    public void dfs(int[] nums,int i,int sum,int tar){
        if(sum == tar){
            res.add(new ArrayList<>(ans));
            return;
        }else if(sum > tar || i == nums.length){
            return;
        }
        for (int j = i; j < nums.length; j++) {
            ans.add(nums[j]);
            dfs(nums,j,sum+nums[j],tar);
            ans.remove(ans.size()-1);
        }
    }
}

/*
https://leetcode.cn/problems/generate-parentheses/?envType=study-plan-v2&envId=top-100-liked
 */


class Solution22_2 {
    List<String> res = new ArrayList<>();
    StringBuilder b = new StringBuilder();
    int len,n;
    public List<String> generateParenthesis(int n) {
        this.n = n;
        dfs(0);
        return res;
    }
    public void dfs(int i){
        if(b.length() == n*2){
            res.add(b.toString());
            return;
        }
        if(b.length() > 0 && i-len < len){
            b.append(')');
            dfs(i+1);

        }else{
            b.append('(');
            len++;
            dfs(i+1);
            len--;
        }
        b.deleteCharAt(b.length()-1);
    }
}

/*
https://leetcode.cn/problems/word-search/description/?envType=study-plan-v2&envId=top-100-liked
 */

class Solution79 {
    int n,m;
    boolean[][] v;
    public boolean exist(char[][] board, String word) {
        n = board.length;
        m = board[0].length;
        v = new boolean[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                boolean t = dfs(board,i,j,word.toCharArray(),0);
                if(t) return true;
            }
        }
        return false;
    }
    public boolean dfs(char[][] b,int i, int j,char[] w,int k){
        if(k == w.length) return true;
        if(i < 0 || i >= n || j < 0 || j >= m || v[i][j]) return false;
        if(b[i][j] == w[k]){
            v[i][j] = true;
            boolean t = dfs(b,i+1,j,w,k+1) || dfs(b,i-1,j,w,k+1) || dfs(b,i,j+1,w,k+1) || dfs(b,i,j-1,w,k+1);
            v[i][j] = false;
            return t;
        }
        return false;
    }
}

/*
https://leetcode.cn/problems/palindrome-partitioning/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution131_2 {
    List<List<String>> ans = new ArrayList<>();
    List<String> chid = new ArrayList<>();
    public List<List<String>> partition(String s) {
        char[] c = s.toCharArray();
        dfs(c,0);
        return ans;
    }
    public void dfs(char[] c,int i){
        if(i== c.length){
            ans.add(new ArrayList<>(chid));
            return;
        }
        for (int j = i; j < c.length; j++) {
            if(is(c,i,j)){
                char[] chars = Arrays.copyOfRange(c, i, j + 1);
                chid.add(new String(chars));
                dfs(c,j+1);
                chid.remove(chid.size()-1);
            }
        }
    }
    public boolean is(char[] c,int i, int j){
        while(i<j){
            if(c[i] != c[j]) return false;
            i++;
            j--;
        }
        return true;
    }
}


/*
https://leetcode.cn/problems/combinations/
 */
class Solution77 {
    List<List<Integer>> ans  = new ArrayList<>();
    List<Integer> chid = new ArrayList<>();
    public List<List<Integer>> combine(int n, int k) {
        dfs(n,1,k);
        return ans;
    }
    public void dfs(int n,int i,int k){
        if(chid.size() == k) {
            ans.add(new ArrayList<>(chid));
            return;
        }
        for(int j = i; j <= n; j++){
            chid.add(j);
            dfs(n,j+1,k);
            chid.remove(chid.size()-1);
        }
    }
}

/*
https://leetcode.cn/problems/n-queens/description/?envType=study-plan-v2&envId=top-100-liked
 */


/*
https://leetcode.cn/problems/find-the-most-competitive-subsequence/description/?envType=daily-question&envId=2024-05-24
 */

public class H100 {
    public static void main(String[] args) {
        Solution1673 solution1673 = new Solution1673();
        int[] ints = solution1673.mostCompetitive(new int[]{3, 5, 2, 6}, 2);
        System.out.println(ints);
    }
}

class Solution1673 {
    //todo 脑袋晕，单调栈，啥时候用，边界条件
    List<List<Integer>> ans = new ArrayList<>();
    public int[] mostCompetitive_v2(int[] nums, int k) {
        int[] st = new int[k];
        int m = 0,n = nums.length;
        for (int i = 0; i < n; i++) {
            int x = nums[i];
            while(m > 0 && x < st[m-1] && m+n-i > k){
                m--;
            }
            if(m < k){
                st[m++] = x;
            }
        }
        return st;
    }


    public int[] mostCompetitive(int[] nums, int k) {
        List<List<Integer>> q = new ArrayList<>(nums.length);
        for (int i = 0; i < nums.length; i++) {
            List<Integer> a = new ArrayList<>();
            a.add(i);
            a.add(nums[i]);
            q.add(a);
        }
        q.sort((o1,o2)->{
            if(o1.get(1).equals(o2.get(1))) return o1.get(0)-o2.get(0);
           return o1.get(1)-o2.get(1);
        });
        dfs(q,k);
        int[] ints = new int[k];
        for (int i = 0; i < ans.size(); i++) {
            ints[i] = ans.get(i).get(1);
        }
        return ints;
        //return ans.stream().mapToInt(Integer::intValue).toArray();
    }
    Map<List<Integer>,Boolean> map = new HashMap<>();
    public void dfs(List<List<Integer>> q,int k){
        if(ans.size() == k) return;
        List<Integer> last = null;
        if(ans.size() > 0) last = ans.get(ans.size() - 1);
        for (int j = 0; j <q.size(); j++) {
            List<Integer> nus = q.get(j);
            if(!map.containsKey(nus)&&
                    (last == null || last.get(0) < nus.get(0))&&
                    nus.get(0)+ (k-ans.size()) <= q.size()){
                ans.add(nus);
                map.put(nus,true);
                dfs(q,k);
                if(ans.size() == k) return;
                ans.remove(ans.size()-1);
                map.remove(nus);
            }
        }

    }
}
