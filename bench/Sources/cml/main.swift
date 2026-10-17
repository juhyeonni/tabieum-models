import CoreML
import Darwin
import Foundation

func footprintMB() -> Double {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}
func residentMB() -> Double {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return Double(info.resident_size) / 1_048_576
}
func line(_ s: String) { print(String(format: "%-46@ footprint %5.0f MB  resident %5.0f MB", s as NSString, footprintMB(), residentMB())) }

let env = ProcessInfo.processInfo.environment
let path = env["PKG"]!, S = Int(env["S"]!)!
let units: MLComputeUnits = ["cpu": .cpuOnly, "gpu": .cpuAndGPU, "ne": .cpuAndNeuralEngine, "all": .all][env["UNITS"] ?? "ne"]!
line("start")
var t = Date()
let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: path))
print(String(format: "compile %.1fs", Date().timeIntervalSince(t)))
let cfg = MLModelConfiguration(); cfg.computeUnits = units
t = Date()
var model: MLModel? = try MLModel(contentsOf: compiled, configuration: cfg)
print(String(format: "load %.1fs", Date().timeIntervalSince(t)))
line("model loaded (\(env["UNITS"] ?? "ne"), S=\(S))")
func input(_ n: Int) throws -> MLDictionaryFeatureProvider {
    let ids = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .int32), mask = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .int32)
    for i in 0..<S { ids[i] = NSNumber(value: i < n ? 100 + i : 1); mask[i] = NSNumber(value: i < n ? 1 : 0) }
    return try MLDictionaryFeatureProvider(dictionary: ["input_ids": ids, "attention_mask": mask])
}
let f = try input(min(S, 40))
t = Date()
_ = try model!.prediction(from: f)
print(String(format: "first prediction %.2fs", Date().timeIntervalSince(t)))
line("after first prediction")
t = Date()
for _ in 0..<200 { _ = try model!.prediction(from: f) }
print(String(format: "steady: %.2f ms/text", Date().timeIntervalSince(t) * 1000 / 200))
line("after 200 predictions")
if let h = env["HOLD"], let sec = Double(h) { print("HOLDING"); fflush(stdout); try await Task.sleep(for: .seconds(sec)) }
model = nil
try await Task.sleep(for: .seconds(1))
line("model released")
