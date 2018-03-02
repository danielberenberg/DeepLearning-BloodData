import basic_utils.basics as base
import basic_utils.video_core as vc

M, header = base.csv2data("./DeepLearningClassData.csv")
print(header)
print(M)

images = vc.video_file_to_frames("data/S0001/Trial1.MOV")

print(images[:10])
