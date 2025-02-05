import torch
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self.activations.append(output)

        def save_gradient(module, input, output):
            self.gradients.append(output[0])

        #target_layer = dict(self.model.named_modules())[self.target_layer]
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)

    def forward(self, data):
        return self.model(data)

    def generate_cam(self, data, device, target_class=None):
        self.activations = []
        self.gradients = []
        output = self.forward(data)
        probs = torch.sigmoid(output).squeeze()

        if target_class is None:
            target_class = probs.argmax(dim=0).item()

        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        gradients = gradients.mean(dim=[0, 2, 3], keepdim=True)
        weighted_activations = activations * gradients
        cam = weighted_activations.mean(dim=1, keepdim=True).squeeze()

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam

    def visualize_cam(self, input_image, cam, alpha=0.5, save_as=None):
        """
        Plots the original image, the Grad-CAM heatmap, and the overlay of the heatmap on the image.

        Parameters:
        - image_tensor (torch.Tensor): The input image tensor (C, H, W) or (1, H, W) that needs to be plotted.
        - gradcam_heatmap (numpy.ndarray): The Grad-CAM heatmap to overlay on the image.
        - alpha (float): Transparency level of the heatmap overlay. Default is 0.5.
        """
        
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_image.shape[1], input_image.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Create the plot with 3 subplots
        fig = plt.figure(figsize=(12, 4))
        
        # Plot the original image
        plt.subplot(1, 3, 1)
        plt.imshow(input_image)
        #plt.title("Original Image")
        plt.axis('off')

        # Plot the heatmap alone
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        #plt.title("Grad-CAM Heatmap")
        plt.axis('off')

        # Plot the original image with the heatmap overlay
        plt.subplot(1, 3, 3)
        plt.imshow(input_image)
        plt.imshow(cam, cmap='jet', alpha=alpha)  # Overlay heatmap with transparency
        #plt.title("Heatmap Overlay")
        plt.axis('off')

        # Display the plot
        plt.subplots_adjust(wspace=0.25, hspace=-0.8, left=0, right=1.0, bottom=0, top=1.0)
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {save_as}")
