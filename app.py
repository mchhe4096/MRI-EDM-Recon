from __future__ import annotations

import gradio as gr

from services.infer_one import reconstruct_for_app


def _protocol_from_checks(show_current_metric: bool, show_paper_metric: bool) -> str:
    if show_current_metric and show_paper_metric:
        return "both"
    if show_current_metric:
        return "current"
    if show_paper_metric:
        return "author"
    raise gr.Error("请至少选择一种指标口径（当前项目口径或论文基线口径）。")


def _run_recon_ui(
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
    num_samples: int,
    show_current_metric: bool,
    show_paper_metric: bool,
    paper_norm: bool,
    paper_eps: float,
):
    if not file_path:
        raise gr.Error("请先上传一个 .pt 文件。")
    if not ckpt_path:
        raise gr.Error("请填写 checkpoint 路径，例如 outputs/ckpts/last.pt")

    protocol = _protocol_from_checks(show_current_metric, show_paper_metric)

    try:
        result = reconstruct_for_app(
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
            eval_protocol=protocol,
            paper_norm=paper_norm,
            paper_eps=paper_eps,
            num_samples=int(num_samples),
        )
    except Exception as e:
        raise gr.Error(f"重建失败: {e}") from e

    return (
        result["summary"],
        result["recon"],
        result["zf"],
        result["gt"],
        result["std"],
        result["samples"],
        result["info"],
    )


def _toggle_blend(mode: str):
    return gr.update(visible=(mode == "blend"))


def _toggle_dc(dc: bool):
    return (
        gr.update(visible=dc),
        gr.update(visible=dc),
        gr.update(visible=dc),
        gr.update(visible=dc),
    )


def _toggle_paper_options(show_paper_metric: bool):
    return gr.update(visible=show_paper_metric), gr.update(visible=show_paper_metric)


def _apply_preset(preset: str):
    # steps, num_samples, dc, init_mode, init_blend, dc_start, dc_every, dc_lam, dc_ramp
    if preset == "快速演示":
        return 20, 1, True, "zf", 0.5, 0.6, 2, 0.15, False
    if preset == "标准质量":
        return 40, 1, True, "zf", 0.5, 0.6, 2, 0.15, True
    if preset == "高质量对比":
        return 60, 3, True, "zf", 0.5, 0.5, 1, 0.18, True
    if preset == "多样性展示":
        return 40, 6, True, "blend", 0.3, 0.6, 2, 0.12, False
    return 40, 1, True, "zf", 0.5, 0.6, 2, 0.15, False


with gr.Blocks(title="MRI 重建封装软件") as demo:
    gr.Markdown("## MRI 重建封装软件")
    gr.Markdown("输入 `.pt`（支持 `k_us+mask` 或 `kspace_full`），输出重建图像并自动对比 ZF 基线。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 基础设置（建议先用默认）")
            file_input = gr.File(label="输入 MRI 切片（.pt）", file_types=[".pt"], type="filepath")
            ckpt_input = gr.Textbox(
                label="Checkpoint 路径",
                value="outputs/ckpts/last.pt",
                placeholder="例如：outputs/ckpts/last.pt",
            )
            preset_input = gr.Radio(
                choices=["快速演示", "标准质量", "高质量对比", "多样性展示"],
                value="标准质量",
                label="参数预设",
            )

            summary_output = gr.Textbox(label="关键结论", lines=2)
            run_btn = gr.Button("开始重建", variant="primary")

            with gr.Accordion("高级参数（实验/对比用）", open=False):
                use_ema_input = gr.Checkbox(label="使用 EMA 权重", value=True)
                include_mask_mode = gr.Radio(
                    choices=["auto", "true", "false"],
                    value="auto",
                    label="include_mask_channel",
                    info="auto 表示跟随 checkpoint 中的训练参数",
                )

                steps_input = gr.Slider(minimum=10, maximum=100, step=1, value=40, label="采样步数")
                num_samples_input = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="采样次数（体现多样性）")
                init_mode_input = gr.Dropdown(
                    choices=["zf", "noise", "blend"],
                    value="zf",
                    label="初始化模式",
                )
                init_blend_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                    label="blend 系数",
                    visible=False,
                )

                dc_input = gr.Checkbox(label="启用数据一致性（DC）", value=True)
                dc_start_input = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.6, label="dc_start", visible=True)
                dc_every_input = gr.Slider(minimum=1, maximum=10, step=1, value=2, label="dc_every", visible=True)
                dc_lam_input = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.15, label="dc_lam", visible=True)
                dc_ramp_input = gr.Checkbox(label="dc_ramp", value=False, visible=True)

                gr.Markdown("#### 指标口径（仅影响打分，不影响图像）")
                show_current_metric_input = gr.Checkbox(label="显示当前项目口径指标", value=True)
                show_paper_metric_input = gr.Checkbox(label="显示论文基线口径指标", value=True)
                paper_norm_input = gr.Checkbox(label="论文口径归一化（paper_norm）", value=True, visible=True)
                paper_eps_input = gr.Number(label="论文口径 eps（高级）", value=1e-8, precision=10, visible=True)

        with gr.Column(scale=1):
            gr.Markdown("### 重建结果")
            recon_output = gr.Image(
                label="重建结果 Recon（完整显示）",
                type="numpy",
                height=420,
            )
            with gr.Row():
                zf_output = gr.Image(label="ZF 基线", type="numpy", height=260)
                gt_output = gr.Image(label="GT（若存在）", type="numpy", height=260)
                std_output = gr.Image(label="样本标准差图", type="numpy", height=260)
            sample_gallery_output = gr.Gallery(
                label="多样本结果画廊（sample_01...）",
                columns=2,
                rows=2,
                height=420,
            )
            info_output = gr.Textbox(label="详细运行信息", lines=14)

    init_mode_input.change(_toggle_blend, inputs=[init_mode_input], outputs=[init_blend_input])
    dc_input.change(
        _toggle_dc,
        inputs=[dc_input],
        outputs=[dc_start_input, dc_every_input, dc_lam_input, dc_ramp_input],
    )
    show_paper_metric_input.change(
        _toggle_paper_options,
        inputs=[show_paper_metric_input],
        outputs=[paper_norm_input, paper_eps_input],
    )

    preset_input.change(
        _apply_preset,
        inputs=[preset_input],
        outputs=[
            steps_input,
            num_samples_input,
            dc_input,
            init_mode_input,
            init_blend_input,
            dc_start_input,
            dc_every_input,
            dc_lam_input,
            dc_ramp_input,
        ],
    )

    run_btn.click(
        _run_recon_ui,
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
            num_samples_input,
            show_current_metric_input,
            show_paper_metric_input,
            paper_norm_input,
            paper_eps_input,
        ],
        outputs=[
            summary_output,
            recon_output,
            zf_output,
            gt_output,
            std_output,
            sample_gallery_output,
            info_output,
        ],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
