package pass_leecode;

import java.util.Arrays;
import java.util.Comparator;

/**
 * @author xwp
 * @date 2023/9/1
 * @Description
 */
public class BucketSort {
    public static void main(String[] args) {
        int i = 0;
        int j = 0;
        System.out.println(1 << j);
//        while(true){
//            i = i|(1<<j);
//            j++;
//        }
    }

    public static void bucketSort(int[] nums){
        int size = nums.length;
        System.out.println(nums[0]);
        swap(nums,0,--size);
        while (size > 0){
            down(nums,0,size);
            System.out.println(nums[0]);
            swap(nums,0,--size);
        }
    }
    public static void sort(int[] nums){
        int len = nums.length;
        for(int i = 0; i < len; i++){
            up(nums,i);
        }
    }
    //上浮
    public static void up(int[] nums, int i) {
        while (nums[i] > nums[(i-1)/2]){
            swap(nums,i,(i-1)/2);
            i = (i-1)/2;
        }
    }
    //下浮
    public static void down(int[] nums, int i, int size){
        int left =  i*2+1;
        while (left < size){
            int largest = left+1 < size && nums[left] < nums[left+1] ? left+1 : left;
            largest = nums[largest] > nums[i] ? largest : i;
            if(largest == i){
                break;
            }else{
                swap(nums,largest,i);
                i = largest;
                left = i*2+1;
            }
        }
    }
    public static void swap(int[] nums,int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
