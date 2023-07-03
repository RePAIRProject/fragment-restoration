
In this experiment, we choose 5 images semantic segmentation network could not ignore black marks and and black marks were included in predictions. 
Then we apply Black mark detection using YOLO on them, 
later we apply our inpainting method, 
and we get semantic segmentation network's prediction on these new "black marks removed" images.
Although, numerical results (meanIOU) remains more or less the same. Visually we can see a more appealing result in cleaned fragments.
One reason for obtaining approximately same numerical results can be the very thin line around the inpainted fragments. This is caused due to constantly changing background color in inpainting process.  

Below we plot the images before and after this method:

Image number 1 before the process:

![1](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_0_cleaned_fragment_added.png)

Image number 1 after the process:

![1](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_0_cleaned_fragment%20_added_inpainted_images.png)

Image number 2 before the process:

![2](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_25_cleaned_fragment_added.png)

Image number 2 after the process:

![2](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_25_cleaned_fragment%20_added_inpainted_images.png)

Image number 3 before the process:

![3](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_2_cleaned_fragment_added.png)

Image number 3 after the process:

![3](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_2_cleaned_fragment%20_added_inpainted_images.png)

Image number 4 before the process:

![4](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_37_cleaned_fragment_added.png)

Image number 4 after the process:

![4](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_37_cleaned_fragment%20_added_inpainted_images.png)

Image number 5 before the process:

![5](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_53_cleaned_fragment_added.png)

Image number 5 after the process:

![5](https://github.com/RePAIRProject/fragment-restoration/blob/develop/effect_of_BlackMarkRemoval_on_semantic_seg/visual_evaluation_test_53_cleaned_fragment%20_added_inpainted_images.png)

