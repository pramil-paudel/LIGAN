import torch
import torch.nn.functional as F
import gan.GanUtils as utils
from stego.Stego_Utlis import compute_metrics, plot_grid_triplet_dual_metrics


def test_step(model_G, model_D, cover_images, container_images, secret_input, gt_secret, batch, batch_size, RESULTS_DIR,
              device, epoch, isValidation):
    RESULTS_TEST_PATH = RESULTS_DIR + ('/validation/' if isValidation else '/test/')
    utils.check_dir([RESULTS_TEST_PATH])

    with torch.no_grad():
        model_G.eval()
        model_D.eval()

        predicted_output = model_G(secret_input.float(), container_images.float())

        # Clamp for evaluation
        norm_cover = torch.clamp(cover_images, 0, 1)
        norm_container = torch.clamp(container_images, 0, 1)
        norm_gt = torch.clamp(gt_secret, 0, 1)
        norm_pred = torch.clamp(predicted_output, 0, 1)

        if epoch % 5 == 0 and batch == 0:
            # Compute metrics between (cover vs container) and (gt vs prediction)
            metrics_cover_vs_container = compute_metrics(norm_cover, norm_container, crop_border=10)
            metrics_gt_vs_pred = compute_metrics(norm_gt, norm_pred, crop_border=0)

            combined_metrics = []
            for m1, m2 in zip(metrics_cover_vs_container, metrics_gt_vs_pred):
                combined_metrics.append((
                    f"Hiding:\nPSNR: {m1['PSNR']:.2f} | SSIM: {m1['SSIM']:.3f}\nMSE: {m1['MSE']:.5f}",
                    f"Reveal:\nPSNR: {m2['PSNR']:.2f} | SSIM: {m2['SSIM']:.3f}\nMSE: {m2['MSE']:.5f}"
                ))

            resize_size = (64, 64)
            vis_cover = F.interpolate(norm_cover, size=resize_size, mode='bilinear', align_corners=False)
            vis_container = F.interpolate(norm_container, size=resize_size, mode='bilinear', align_corners=False)
            vis_secret_in = F.interpolate(secret_input, size=resize_size, mode='bilinear', align_corners=False)
            vis_predicted = F.interpolate(norm_pred, size=resize_size, mode='bilinear', align_corners=False)
            vis_gt = F.interpolate(norm_gt, size=resize_size, mode='bilinear', align_corners=False)

            save_path = f"{RESULTS_TEST_PATH}validation_epoch_{epoch}.png"
            plot_grid_triplet_dual_metrics(
                cover_out=vis_cover[:4],
                container_out=vis_container[:4],
                secret_out=vis_secret_in[:4],
                reco_out=vis_predicted[:4],
                original_gt=vis_gt[:4],
                metrics=combined_metrics,
                num_samples=4,
                save_path=save_path
            )
            print(f"Saved visualization with metrics: {save_path}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return norm_pred
