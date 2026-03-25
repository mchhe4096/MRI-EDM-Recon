from __future__ import annotations

import gradio as gr

from services.infer_one import reconstruct_from_pt


def _run_recon(
    file_path: str,
    ckpt_path: str,
    steps: int,
    dc: bool,
    init_mode: str,
    init_blend: float,
    dc_start: float,
    dc_every: int,
    dc_lam: float,
    dc_ramp: bool,
    use_ema: bool,
    include_mask_channel_mode: str,
):
    if not file_path:
        raise gr.Error("请先上传一个 .pt 文件。")
    if not ckpt_path:
        raise gr.Error("请填写 checkpoint 路径（例如 outputs/ckpts/last.pt）。")

    try:
        recon, zf, gt, info = reconstruct_from_pt(
            pt_path=file_path,
            ckpt_path=ckpt_path,
            steps=steps,
            dc=dc,
            init_mode=init_mode,
            init_blend=init_blend,
            dc_start=dc_start,
            dc_every=dc_every,
            dc_lam=dc_lam,
            dc_ramp=dc_ramp,
            use_ema=use_ema,
            include_mask_channel_mode=include_mask_channel_mode,
        )
    except Exception as e:
        raise gr.Error(f"重建失败: {e}") from e

    return recon, zf, gt, info


def _toggle_blend(mode: str):
    return gr.update(visible=(mode == "blend"))


def _toggle_dc(dc: bool):
    return (
        gr.update(visible=dc),
        gr.update(visible=dc),
        gr.update(visible=dc),
        gr.update(visible=dc),
    )


with gr.Blocks(title="MRI-EDM-Recon Demo") as demo:
    gr.Markdown("## MRI 重建演示（Gradio）")
    gr.Markdown("上传单个 `.pt` 切片（需包含 `kspace_full`），输出重建后的 MRI 图像。")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="输入 MRI 切片 (.pt)", file_types=[".pt"], type="filepath")
            ckpt_input = gr.Textbox(
                label="Checkpoint 路径",
                value="outputs/ckpts/last.pt",
                placeholder="例如: outputs/ckpts/last.pt",
            )
            use_ema_input = gr.Checkbox(label="使用 EMA 权重", value=True)
            include_mask_mode = gr.Radio(
                choices=["auto", "true", "false"],
                value="auto",
                label="include_mask_channel",
                info="auto=跟随checkpoint训练参数",
            )

            steps_input = gr.Slider(minimum=10, maximum=100, step=1, value=40, label="采样步数 steps")
            dc_input = gr.Checkbox(label="启用数据一致性 DC", value=True)
            init_mode_input = gr.Dropdown(
                choices=["zf", "noise", "blend"],
                value="zf",
                label="采样初始化 init_mode",
            )
            init_blend_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.5,
                label="blend 系数 init_blend",
                visible=False,
            )

            dc_start_input = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.6, label="dc_start", visible=True)
            dc_every_input = gr.Slider(minimum=1, maximum=10, step=1, value=2, label="dc_every", visible=True)
            dc_lam_input = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.15, label="dc_lam", visible=True)
            dc_ramp_input = gr.Checkbox(label="dc_ramp", value=False, visible=True)

            run_btn = gr.Button("开始重建", variant="primary")

        with gr.Column(scale=1):
            recon_output = gr.Image(label="重建结果 Recon", type="numpy")
            zf_output = gr.Image(label="零填充重建 ZF", type="numpy")
            gt_output = gr.Image(label="GT（若输入含 img_gt）", type="numpy")
            info_output = gr.Textbox(label="运行信息", lines=8)

    init_mode_input.change(_toggle_blend, inputs=[init_mode_input], outputs=[init_blend_input])
    dc_input.change(
        _toggle_dc,
        inputs=[dc_input],
        outputs=[dc_start_input, dc_every_input, dc_lam_input, dc_ramp_input],
    )

    run_btn.click(
        _run_recon,
        inputs=[
            file_input,
            ckpt_input,
            steps_input,
            dc_input,
            init_mode_input,
            init_blend_input,
            dc_start_input,
            dc_every_input,
            dc_lam_input,
            dc_ramp_input,
            use_ema_input,
            include_mask_mode,
        ],
        outputs=[recon_output, zf_output, gt_output, info_output],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

