package lc;

import java.util.*;

import static java.util.Objects.hash;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}


public class Deep {
    public static void main(String[] args) {
        String s = "123";
        String s2 = "123";
        String s3 = new String("123");
        String s4 = new String("123");
        System.out.println(hash(s.hashCode()));
        System.out.println(hash(s4));
        System.out.println(hash(s3));
    }
    class Node {
        TreeNode node;
        int deep;

        Node(TreeNode node, int de) {
            this.node = node;
            this.deep = de;
        }
    }

    public TreeNode lcaDeepestLeaves(TreeNode root) {
        Node p = to(root, 0);
        return p.node;
    }


    public Node to(TreeNode root, int de) {
        if (root == null) return null;
        if (root.left == null && root.right == null) return new Node(root, de);
        Node p = to(root.left, de + 1);
        Node q = to(root.right, de + 1);
        if (p != null && q != null) {
            if (p.deep == q.deep) return new Node(root, p.deep);
            return p.deep > q.deep ? p : q;
        }
        if (p != null) return p;
        if (q != null) return q;
        return new Node(root, de);
    }

}

class Solution131 {
    List<List<String>> ans = new ArrayList<>();
    List<String> r = new ArrayList<>();

    public List<List<String>> partition(String s) {
        dfs(s, 0);
        return ans;
    }

    public void dfs(String s, int i) {
        if (i == s.length()) {
            ans.add(new ArrayList<>(r));
            return;
        }
        for (int j = i; j < s.length(); j++) {
            if (is(s, i, j)) {
                r.add(s.substring(i, j + 1));
                dfs(s, j + 1);
                r.remove(r.size() - 1);
            }
        }
    }

    private boolean is(String s, int i, int j) {
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j++)) return false;
        }
        return true;
    }


    public void dfs2(String s, int i) {
        if (i == s.length()) {
            ans.add(new ArrayList<>(r));
            return;
        }
        for (int j = i; j < s.length(); j++) {
            if (is(s, i, j)) {
                r.add(s.substring(i, j + 1));
                dfs(s, j + 1);
                r.remove(r.size() - 1);
            }
        }
    }
}

class Solution1593 {
    int ans = 0;
    Set<String> set = new HashSet<>();

    public int maxUniqueSplit(String s) {
        dfs(s, 0);
        return ans;
    }

    public void dfs(String s, int i) {
        if (i == s.length()) {
            ans = Math.max(ans, set.size());
            return;
        }
        for (int j = i; j < s.length(); j++) {
            String child = s.substring(i, j + 1);
            if (!set.contains(child)) {
                set.add(child);
                dfs(s, j + 1);
                set.remove(child);
            }
        }
    }

}


class RUNClass{
    public static void main(String[] args) {
        Solution22 solution22 = new Solution22();
        List<String> strings = solution22.generateParenthesis(3);

    }
}
/**
 * 22 括号生成
 * https://leetcode.cn/problems/generate-parentheses/description/
 */
//todo 不会
class Solution22 {

    private int n = 0;
    List<String> ans = new ArrayList<>();
    char[] path = null;
    //枚举看啥时候填括号
    public List<String> generateParenthesis(int n) {
        this.n = n;
        path = new char[n*2];
        dfs(0,0);
        return ans;
    }
    public void dfs(int i,int open){
        if (i == n*2){
            ans.add(new String(path));
            return;
        }
        if(open < n){
            path[i] = '(';
            dfs(i+1,open+1);
        }
        if(i-open < open){
            path[i] = ')';
            dfs(i+1,open);
        }
    }
}

/**
 * 46 全排列
 * https://leetcode.cn/problems/permutations/description
 */
class Solution46 {
    List<List<Integer>> ans = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    boolean[] map = new boolean[30];
    int len;
    public List<List<Integer>> permute(int[] nums) {
        len = nums.length;
        dfs(nums);
        return ans;
    }
    public void dfs(int[] nums){
        if(path.size() == len){
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int j = 0; j < len; j++) {
            if(!map[nums[j]+10]){
                path.add(nums[j]);
                map[nums[j]+10] = true;
                dfs(nums);
                map[nums[j]+10] = false;
                path.remove(path.size()-1);
            }
        }

    }
}

/**
 * n皇后
 * https://leetcode.cn/problems/n-queens/
 */
class Solution51 {
    List<List<String>> ans = new ArrayList<>();
    List<String> ansPath = new ArrayList<>();
    char[][] path = null;
    int n;
    public List<List<String>> solveNQueens(int n) {
        this.n = n;
        path = new char[n][n];
        for (char[] chars : path) {
            Arrays.fill(chars, '.');
        }
        dfs(0);
        return ans;
    }
    public void dfs(int i){
        if(i == n){
            for (int i1 = 0; i1 < path.length; i1++) {
                ansPath.add(new String(path[i1]));
            }
            ans.add(new ArrayList<>(ansPath));
            ansPath.clear();
        }
        for (int j = 0; j < n; j++) {
            if(isValid(i,j)){
                path[i][j] = 'Q';
                dfs(i+1);
                path[i][j] = '.';
            }
        }
    }
    public boolean isValid(int i,int j){
        for (int k = i-1; k >= 0; k--) {
            if (path[k][j] == 'Q') return false;
        }
        for (int k = i-1, p = j+1; k >= 0 && p < n; k--,p++) {
            if (path[k][p] == 'Q') return false;
        }
        for (int k = i-1,p = j-1; k >= 0 && p >= 0 ; k--,p--) {
            if (path[k][p] == 'Q') return false;
        }
        return true;

    }

}

/**
 * n皇后2
 * https://leetcode.cn/problems/n-queens-ii/description/
 */
class Solution52 {
    char[][] path = null;
    int n,ans;
    public int totalNQueens(int n) {
        this.n = n;
        path = new char[n][n];
        for (char[] chars : path) {
            Arrays.fill(chars, '.');
        }
        dfs(0);
        return ans;
    }
    public void dfs(int i){
        if(i == n){
            ans++;
        }
        for (int j = 0; j < n; j++) {
            if(isValid(i,j)){
                path[i][j] = 'Q';
                dfs(i+1);
                path[i][j] = '.';
            }
        }
    }
    public boolean isValid(int i,int j){
        for (int k = i-1; k >= 0; k--) {
            if (path[k][j] == 'Q') return false;
        }
        for (int k = i-1, p = j+1; k >= 0 && p < n; k--,p++) {
            if (path[k][p] == 'Q') return false;
        }
        for (int k = i-1,p = j-1; k >= 0 && p >= 0 ; k--,p--) {
            if (path[k][p] == 'Q') return false;
        }
        return true;

    }

}


