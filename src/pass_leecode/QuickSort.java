package pass_leecode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 快排！
 * @author xwp
 * @date 2023/7/8
 * @Description
 */
public class QuickSort {
    public static void main(String[] args) {
        System.out.println(Integer.numberOfTrailingZeros(4));
        String s = "aoienf";
        boolean oi = s.contains("oi");
        System.out.println(oi);

    }

    public static void sort(int[] nums, int l ,int r){
        int i = l, j = r;
        if(i >= j) return;
        while(i < j){
            while(i<j && nums[j] <= nums[i]){
                j--;
            }
            if(i<j){
                swap(nums,i,j);
                i++;
            }
            while(i<j && nums[i] >= nums[j]){
                i++;
            }
            if(i<j){
                swap(nums,i,j);
                j--;
            }
        }
        sort(nums,l,i);
        sort(nums,i+1,r);
    }

    public static void swap(int[] nums, int i, int j){
        int t = nums[j];
        nums[j] = nums[i];
        nums[i] = t;
    }


}
