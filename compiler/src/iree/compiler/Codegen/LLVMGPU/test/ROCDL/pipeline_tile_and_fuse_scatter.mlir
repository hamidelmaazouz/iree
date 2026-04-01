// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-executable-target{for-rocdl=true}))" %s | FileCheck %s

// TODO: This test case fails with the module-scope pipeline because
// VerifyWorkgroupDistributionPass rejects a vector.transfer_write to global
// memory that ends up outside the distributed scf.forall context after
// scatter fusion lowering. Needs a fix in the scatter distribution path.
// Once fixed, migrate this test back to pipeline_tile_and_fuse.mlir with the
// module-scope pipeline.

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [64, 1, 1] subgroup_size = 64>
#lowering_config = #iree_gpu.lowering_config<{thread = [1, 1, 1, 4], workgroup = [1, 1, 32, 8]}>

#pipeline_layout = #hal.pipeline.layout<bindings =
            [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
            #hal.pipeline.binding<storage_buffer, ReadOnly>,
            #hal.pipeline.binding<storage_buffer, Indirect>,
            #hal.pipeline.binding<storage_buffer, Indirect>
]>

func.func @multi_result_index_generic_with_scatter_fusion() attributes {
    hal.executable.target = #executable_target_rocm,
    translation_info = #translation_info} {
  %c32_i64 = arith.constant 32 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %24 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1) flags("ReadOnly|Indirect") : memref<4x?x32x8xf16, #hal.descriptor_type<storage_buffer>>{%c1}
  %25 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c1) flags("ReadOnly|Indirect") : memref<4x?xi64, #hal.descriptor_type<storage_buffer>>{%c1}
  %26 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<?x8x32xf8E4M3FNUZ, #hal.descriptor_type<storage_buffer>>{%c1}
  %27 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c1) flags(Indirect) : memref<4x?x32x8xf8E4M3FNUZ, strided<[?, 256, 8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%c1}
  %28 = iree_codegen.load_from_buffer %24 : memref<4x?x32x8xf16, #hal.descriptor_type<storage_buffer>> -> tensor<4x?x32x8xf16>
  %29 = iree_codegen.load_from_buffer %25 : memref<4x?xi64, #hal.descriptor_type<storage_buffer>> -> tensor<4x?xi64>
  %30 = tensor.empty(%c1) : tensor<?x8x32xf8E4M3FNUZ>
  %31 = tensor.empty(%c1) : tensor<4x?x8x32xf8E4M3FNUZ>
  %32 = tensor.empty(%c1) : tensor<4x?x32x8xf8E4M3FNUZ>
  %33:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<4x?x32x8xf16>) outs(%32, %31 : tensor<4x?x32x8xf8E4M3FNUZ>, tensor<4x?x8x32xf8E4M3FNUZ>) attrs =  {lowering_config = #lowering_config} {
  ^bb0(%in: f16, %out: f8E4M3FNUZ, %out_0: f8E4M3FNUZ):
    %35 = linalg.index 1 : index
    %36 = arith.mulf %in, %in : f16
    %37 = arith.cmpi eq, %35, %c0 : index
    %38 = arith.select %37, %36, %in : f16
    %39 = arith.truncf %38 : f16 to f8E4M3FNUZ
    linalg.yield %39, %39 : f8E4M3FNUZ, f8E4M3FNUZ
  } -> (tensor<4x?x32x8xf8E4M3FNUZ>, tensor<4x?x8x32xf8E4M3FNUZ>)
  %34 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%33#1, %29 : tensor<4x?x8x32xf8E4M3FNUZ>, tensor<4x?xi64>) outs(%30 : tensor<?x8x32xf8E4M3FNUZ>) {
  ^bb0(%arg0: f8E4M3FNUZ, %arg1: f8E4M3FNUZ):
    iree_linalg_ext.yield %arg0 : f8E4M3FNUZ
  } -> tensor<?x8x32xf8E4M3FNUZ>
  iree_codegen.store_to_buffer %34, %26 : tensor<?x8x32xf8E4M3FNUZ> into memref<?x8x32xf8E4M3FNUZ, #hal.descriptor_type<storage_buffer>>
  iree_codegen.store_to_buffer %33#0, %27 : tensor<4x?x32x8xf8E4M3FNUZ> into memref<4x?x32x8xf8E4M3FNUZ, strided<[?, 256, 8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  return
}
// CHECK-LABEL: func @multi_result_index_generic_with_scatter_fusion
// CHECK:         scf.forall (%arg0, %arg1) in (4, 1) {
// CHECK:           vector.transfer_read
// CHECK:           arith.mulf {{.*}} vector<4xf16>
// CHECK:           arith.truncf {{.*}} vector<4xf16> to vector<4xf8E4M3FNUZ>
// CHECK:           vector.transfer_write
// CHECK:           vector.transfer_write
// CHECK:           iree_linalg_ext.scatter
