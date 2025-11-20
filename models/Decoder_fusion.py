#原 融合阶段无注意力
    # def forward(self, pre_features, post_features):

    #     pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

    #     post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

    #     '''
    #         Stage I  t1，t2，t3时刻特征三种拼接方式，分别放入VSS模块中
    #     '''
    #     p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
    #     B, C, H, W = pre_feat_4.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
    #     ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
    #     p42 = self.st_block_42(ct_tensor_42)

    #     ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_43[:, :, :, 0:W] = pre_feat_4
    #     ct_tensor_43[:, :, :, W:] = post_feat_4
    #     p43 = self.st_block_43(ct_tensor_43)

    #     p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
    #     print(p4.size())    #([8, 128, 8, 8])

    #     '''
    #         Stage II
    #     '''
    #     p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
    #     B, C, H, W = pre_feat_3.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
    #     ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
    #     p32 = self.st_block_32(ct_tensor_32)

    #     ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_33[:, :, :, 0:W] = pre_feat_3
    #     ct_tensor_33[:, :, :, W:] = post_feat_3
    #     p33 = self.st_block_33(ct_tensor_33)

    #     p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
    #     p3 = self._upsample_add(p4, p3)
    #     p3 = self.smooth_layer_3(p3)
    #     print(p3.size())#([8, 128, 16, 16])
    #     '''
    #         Stage III
    #     '''
    #     p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
    #     B, C, H, W = pre_feat_2.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
    #     ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
    #     p22 = self.st_block_22(ct_tensor_22)

    #     ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_23[:, :, :, 0:W] = pre_feat_2
    #     ct_tensor_23[:, :, :, W:] = post_feat_2
    #     p23 = self.st_block_23(ct_tensor_23)

    #     p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
    #     p2 = self._upsample_add(p3, p2)
    #     p2 = self.smooth_layer_2(p2)
    #     print(p2.size())  #([8, 128, 32, 32])
    #     '''
    #         Stage IV
    #     '''
    #     p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
    #     B, C, H, W = pre_feat_1.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
    #     ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
    #     p12 = self.st_block_12(ct_tensor_12)

    #     ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_13[:, :, :, 0:W] = pre_feat_1
    #     ct_tensor_13[:, :, :, W:] = post_feat_1
    #     p13 = self.st_block_13(ct_tensor_13)

    #     p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))
    #     print(p1.size())   #torch.Size([8, 128, 64, 64])
    #     p1 = self._upsample_add(p2, p1)
    #     print("after upsampling:",p1.size()) #torch.Size([8, 128, 64, 64])
    #     p1 = self.smooth_layer_1(p1)   
    #     print("after RESNET:",p1.size()) #torch.Size([8, 128, 64, 64])
    #     return p1

# 1   p4与p3融合，p2与p1融合，融合后再次融合，并用spatial attention 
    # def forward(self, pre_features, post_features):

    #     pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

    #     post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

    #     '''
    #         Stage I  t1，t2，t3时刻特征三种拼接方式，分别放入VSS模块中
    #     '''
    #     p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))  #128,w
    #     #print(p41.size())
    #     B, C, H, W = pre_feat_4.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
    #     ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
    #     p42 = self.st_block_42(ct_tensor_42)  
    #     #print(p42.size())                              #128,2w

    #     ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_43[:, :, :, 0:W] = pre_feat_4
    #     ct_tensor_43[:, :, :, W:] = post_feat_4
    #     p43 = self.st_block_43(ct_tensor_43)
    #     #print(p43.size())

    #     p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
    #     #print(p4.size())

    #     '''
    #         Stage II
    #     '''
    #     p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
    #     B, C, H, W = pre_feat_3.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
    #     ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
    #     p32 = self.st_block_32(ct_tensor_32)

    #     ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_33[:, :, :, 0:W] = pre_feat_3
    #     ct_tensor_33[:, :, :, W:] = post_feat_3
    #     p33 = self.st_block_33(ct_tensor_33)

    #     p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
    #     p3 = self._upsample_add(p4, p3)
    #     p3 = self.smooth_layer_3(p3)
       
    #     '''
    #         Stage III
    #     '''
    #     p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
    #     B, C, H, W = pre_feat_2.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
    #     ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
    #     p22 = self.st_block_22(ct_tensor_22)

    #     ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_23[:, :, :, 0:W] = pre_feat_2
    #     ct_tensor_23[:, :, :, W:] = post_feat_2
    #     p23 = self.st_block_23(ct_tensor_23)

    #     p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
    #     #p2 = self._upsample_add(p3, p2)
    #     #p2 = self.smooth_layer_2(p2)
       
    #     '''
    #         Stage IV
    #     '''
    #     p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
    #     B, C, H, W = pre_feat_1.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
    #     ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
    #     p12 = self.st_block_12(ct_tensor_12)

    #     ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_13[:, :, :, 0:W] = pre_feat_1
    #     ct_tensor_13[:, :, :, W:] = post_feat_1
    #     p13 = self.st_block_13(ct_tensor_13)

    #     p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

    #     p1 = self._upsample_add(p2, p1)
    #     #p1 = self.smooth_layer_1(p1)
    #     #print(p1.size())  #torch.Size([8, 128, 64, 64])     
    #     p3 = self._upsample_add(p3, p1)
    #     #print(p3.size()) #torch.Size([8, 128, 64, 64])
    #     p1 = self.fusion3(p1,p3)


    #     return p11
#2   只有spatial注意力
    # def _upsample_(self, x, y):
    #     _, _, H, W = y.size()
    #     return F.interpolate(x, size=(H, W), mode='bilinear') 
    
    # def forward(self, pre_features, post_features):

    #     pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

    #     post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

    #     '''
    #         Stage I  t1，t2，t3时刻特征三种拼接方式，分别放入VSS模块中
    #     '''
    #     p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
    #     B, C, H, W = pre_feat_4.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
    #     ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
    #     p42 = self.st_block_42(ct_tensor_42)

    #     ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_43[:, :, :, 0:W] = pre_feat_4
    #     ct_tensor_43[:, :, :, W:] = post_feat_4
    #     p43 = self.st_block_43(ct_tensor_43)

    #     p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
    #     #print(p4.size())    #([8, 128, 8, 8])

    #     '''
    #         Stage II
    #     '''
    #     p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
    #     B, C, H, W = pre_feat_3.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
    #     ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
    #     p32 = self.st_block_32(ct_tensor_32)

    #     ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_33[:, :, :, 0:W] = pre_feat_3
    #     ct_tensor_33[:, :, :, W:] = post_feat_3
    #     p33 = self.st_block_33(ct_tensor_33)

    #     p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
    #     p4 = self._upsample_(p4, p3)
    #     #print(p4.size())
    #     #print(p3.size())
    #     p33 = self.fusion3(p4,p3)
    #     #print(p33.size())#([8, 128, 16, 16])
    #     '''
    #         Stage III
    #     '''
    #     p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
    #     B, C, H, W = pre_feat_2.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
    #     ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
    #     p22 = self.st_block_22(ct_tensor_22)

    #     ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_23[:, :, :, 0:W] = pre_feat_2
    #     ct_tensor_23[:, :, :, W:] = post_feat_2
    #     p23 = self.st_block_23(ct_tensor_23)

    #     p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
    #     p33 = self._upsample_(p33, p2)
    #     p22 = self.fusion3(p2,p33)
    #     #print(p2.size())  #([8, 128, 32, 32])
    #     '''
    #         Stage IV
    #     '''
    #     p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
    #     B, C, H, W = pre_feat_1.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
    #     # Fill in odd columns with A and even columns with B
    #     ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
    #     ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
    #     p12 = self.st_block_12(ct_tensor_12)

    #     ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
    #     ct_tensor_13[:, :, :, 0:W] = pre_feat_1
    #     ct_tensor_13[:, :, :, W:] = post_feat_1
    #     p13 = self.st_block_13(ct_tensor_13)

    #     p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))
    #     #print(p1.size())   #torch.Size([8, 128, 64, 64])
    #     p22 = self._upsample_(p22, p1)
    #     #print("after upsampling:",p1.size()) #torch.Size([8, 128, 64, 64])
    #     p11 = self.fusion3(p1,p22)   
    #     #print("after RESNET:",p1.size()) #torch.Size([8, 128, 64, 64])
    #     return p11


# #3 最终版 三种时空融合＋CBAM
#     def _upsample_(self, x, y):
#         _, _, H, W = y.size()
#         return F.interpolate(x, size=(H, W), mode='bilinear') 
#     def forward(self, pre_features, post_features):

#         pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

#         post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

#         '''
#             Stage I  t1，t2，t3时刻特征三种拼接方式，分别放入VSS模块中
#         '''
#         p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
#         B, C, H, W = pre_feat_4.size()
#         # Create an empty tensor of the correct shape (B, C, H, 2*W)
#         ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
#         # Fill in odd columns with A and even columns with B
#         ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
#         ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
#         p42 = self.st_block_42(ct_tensor_42)

#         ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
#         ct_tensor_43[:, :, :, 0:W] = pre_feat_4
#         ct_tensor_43[:, :, :, W:] = post_feat_4
#         p43 = self.st_block_43(ct_tensor_43)

#         p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
#         #print(p4.size())    #([8, 128, 8, 8])

#         '''
#             Stage II
#         '''
#         p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
#         B, C, H, W = pre_feat_3.size()
#         # Create an empty tensor of the correct shape (B, C, H, 2*W)
#         ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
#         # Fill in odd columns with A and even columns with B
#         ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
#         ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
#         p32 = self.st_block_32(ct_tensor_32)

#         ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
#         ct_tensor_33[:, :, :, 0:W] = pre_feat_3
#         ct_tensor_33[:, :, :, W:] = post_feat_3
#         p33 = self.st_block_33(ct_tensor_33)

#         p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
#         p4 = self._upsample_(p4, p3)
#         #print(p4.size())
#         #print(p3.size())
#         p33 = self.fusion4(p4,p3)
#         #print(p33.size())#([8, 128, 16, 16])
#         '''
#             Stage III
#         '''
#         p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
#         B, C, H, W = pre_feat_2.size()
#         # Create an empty tensor of the correct shape (B, C, H, 2*W)
#         ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
#         # Fill in odd columns with A and even columns with B
#         ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
#         ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
#         p22 = self.st_block_22(ct_tensor_22)

#         ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
#         ct_tensor_23[:, :, :, 0:W] = pre_feat_2
#         ct_tensor_23[:, :, :, W:] = post_feat_2
#         p23 = self.st_block_23(ct_tensor_23)

#         p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
#         p33 = self._upsample_(p33, p2)
#         p22 = self.fusion4(p2,p33)
#         #print(p2.size())  #([8, 128, 32, 32])
#         '''
#             Stage IV
#         '''
#         p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
#         B, C, H, W = pre_feat_1.size()
#         # Create an empty tensor of the correct shape (B, C, H, 2*W)
#         ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
#         # Fill in odd columns with A and even columns with B
#         ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
#         ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
#         p12 = self.st_block_12(ct_tensor_12)

#         ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
#         ct_tensor_13[:, :, :, 0:W] = pre_feat_1
#         ct_tensor_13[:, :, :, W:] = post_feat_1
#         p13 = self.st_block_13(ct_tensor_13)

#         p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))
#         #print(p1.size())   #torch.Size([8, 128, 64, 64])
#         p22 = self._upsample_(p22, p1)
#         #print("after upsampling:",p1.size()) #torch.Size([8, 128, 64, 64])
#         p11 = self.fusion4(p1,p22)   
#         #print("after RESNET:",p1.size()) #torch.Size([8, 128, 64, 64])
#         return p11

#4  消融实验：时空融合只有顺序融合
    # def _upsample_(self, x, y):
    #     _, _, H, W = y.size()
    #     return F.interpolate(x, size=(H, W), mode='bilinear') 
    # def forward(self, pre_features, post_features):

    #     pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

    #     post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

    #     '''
    #         Stage I  t1，t2，t3时刻特征三种拼接方式，分别放入VSS模块中
    #     '''
    #     p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
    #     B, C, H, W = pre_feat_4.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     # ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
    #     # # Fill in odd columns with A and even columns with B
    #     # ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
    #     # ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
    #     # p42 = self.st_block_42(ct_tensor_42)

    #     # ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
    #     # ct_tensor_43[:, :, :, 0:W] = pre_feat_4
    #     # ct_tensor_43[:, :, :, W:] = post_feat_4
    #     # p43 = self.st_block_43(ct_tensor_43)

    #     #p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
    #     p4 = self.fuse1_layer_4(p41)
    #     #print(p4.size())    #([8, 128, 8, 8])

    #     '''
    #         Stage II
    #     '''
    #     p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
    #     B, C, H, W = pre_feat_3.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     # ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
    #     # # Fill in odd columns with A and even columns with B
    #     # ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
    #     # ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
    #     # p32 = self.st_block_32(ct_tensor_32)

    #     # ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
    #     # ct_tensor_33[:, :, :, 0:W] = pre_feat_3
    #     # ct_tensor_33[:, :, :, W:] = post_feat_3
    #     # p33 = self.st_block_33(ct_tensor_33)

    #     # p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
    #     p3= self.fuse1_layer_3(p31)
    #     p4 = self._upsample_(p4, p3)
    #     # #print(p4.size())
    #     # #print(p3.size())
    #     p33 = self.fusion3(p4,p3)

    #     #print(p33.size())#([8, 128, 16, 16])
    #     '''
    #         Stage III
    #     '''
    #     p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
    #     B, C, H, W = pre_feat_2.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     # ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
    #     # # Fill in odd columns with A and even columns with B
    #     # ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
    #     # ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
    #     # p22 = self.st_block_22(ct_tensor_22)

    #     # ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
    #     # ct_tensor_23[:, :, :, 0:W] = pre_feat_2
    #     # ct_tensor_23[:, :, :, W:] = post_feat_2
    #     # p23 = self.st_block_23(ct_tensor_23)

    #     # p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
    #     p2 = self.fuse1_layer_2(p21)
    #     p33 = self._upsample_(p33, p2)
    #     p22 = self.fusion3(p2,p33)
    #     #print(p2.size())  #([8, 128, 32, 32])
    #     '''
    #         Stage IV
    #     '''
    #     p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
    #     B, C, H, W = pre_feat_1.size()
    #     # Create an empty tensor of the correct shape (B, C, H, 2*W)
    #     # ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
    #     # # Fill in odd columns with A and even columns with B
    #     # ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
    #     # ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
    #     # p12 = self.st_block_12(ct_tensor_12)

    #     # ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
    #     # ct_tensor_13[:, :, :, 0:W] = pre_feat_1
    #     # ct_tensor_13[:, :, :, W:] = post_feat_1
    #     # p13 = self.st_block_13(ct_tensor_13)

    #     # p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))
    #     # #print(p1.size())   #torch.Size([8, 128, 64, 64]
    #     p1 = self.fuse1_layer_1(p11)
    #     p22 = self._upsample_(p22, p1)
    #     #print("after upsampling:",p1.size()) #torch.Size([8, 128, 64, 64])
    #     p11 = self.fusion4(p1,p22)   
    #     #print("after RESNET:",p1.size()) #torch.Size([8, 128, 64, 64])
    #     return p11

# #交叉注意力＋CBAM
#     def _upsample_(self, x, y):
#         _, _, H, W = y.size()
#         return F.interpolate(x, size=(H, W), mode='bilinear') 
    
#     def forward(self, pre_features, post_features):
        
#         # 选择设备
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # 将所有特征图移动到指定设备
#         pre_features = [feat.to(device) for feat in pre_features]
#         post_features = [feat.to(device) for feat in post_features]

#         pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

#         post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features
#         # print(pre_feat_4.size())# torch.Size([4, 768, 8, 8])
#         # print(pre_feat_3.size())#torch.Size([4, 384, 16, 16])
#         # print(pre_feat_2.size())# torch.Size([4, 192, 32, 32])
#         # print(pre_feat_1.size())# torch.Size([4, 96, 64, 64])

#         '''
#             Stage I 
#         '''
#         # print(pre_feat_4.size()) #torch.Size([4, 768, 8, 8])
#        # p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
#         #print(p41.size())  #torch.Size([4, 128, 8, 8])
#         B, C, H, W = pre_feat_4.size()
#         B, C, H, W = pre_feat_4.size()
#         pre_feat_4 =self.conv_768(pre_feat_4)
#         post_feat_4 = self.conv_768(post_feat_4)
#         p4 = self.crossattention4(pre_feat_4, pre_feat_4)
#         #print(p4.size())#torch.Size([4, 128, 8, 8])
#         '''
#             Stage II
#         '''
#         pre_feat_3 = self.conv_384(pre_feat_3)
#         post_feat_3 = self.conv_384(post_feat_3)
#         #print("pre 3:", pre_feat_3.size())#[4,128,16,16]
#         p3= self.crossattention3(pre_feat_3, post_feat_3) #[4,128,16,16]
#         #print("p3",p3.size())

#         p4 = self._upsample_(p4, p3)
#         #print(p4.size())
#         #print(p3.size())
#         p33 = self.fusion4(p4,p3)
#         #print("p33",p33.size())#([4, 128, 16, 16])     

#         '''
#             Stage III
#         '''
#         pre_feat_2 = self.conv_192(pre_feat_2)
#         post_feat_2 = self.conv_192(post_feat_2)
#         #print("pre 2:", pre_feat_2.size())#[4,128,32,32]
#         p2= self.crossattention2(pre_feat_2, post_feat_2)
#         #print(p2.size())

#         p33 = self._upsample_(p33, p2)
#         #print(p4.size())
#         #print(p3.size())
#         p22 = self.fusion4(p33,p2)
#         #print("p22",p22.size())#([8, 128, 32,32])     

#         '''
#             Stage Ⅳ
#         '''
#         pre_feat_1 = self.conv_96(pre_feat_1)
#         post_feat_1 = self.conv_96(post_feat_1)
#         #print("pre 1:", pre_feat_1.size())#[4,128,64,64]
#         p1= self.crossattention1(pre_feat_1, post_feat_1)
#         #print(p1.size())#[4,128,64,64]

#         p22 = self._upsample_(p22, p1)
#         #print(p4.size())
#         #print(p3.size())
#         p11 = self.fusion4(p22,p1)
#         #print("p11",p11.size())#[4,128,64,64]
#         return p11

# """
# 含有SRA的交叉注意力:注意力分块
# """ 
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio= 8):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wk = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)

#             self.norm1 = nn.LayerNorm(dim)
#             # self.sr1 = nn.Sequential(
#             #     nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio),
#             #     nn.BatchNorm2d(dim),  # 比 LayerNorm 更轻量
#             #     nn.GELU()
#             # )
#             self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)

#             self.norm2 = nn.LayerNorm(dim)

"""
含有SRA的交叉注意力
""" 
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wk = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)

#     def forward(self, x1, x2):
#         B1, C1, H1, W1 = x1.shape
#         B2, C2, H2, W2 = x2.shape

#         # 确保输入特征的通道数与 dim 一致
#         assert C1 == self.wq.in_channels, "x1 的通道数必须等于 dim"
#         assert C2 == self.wk.in_channels, "x2 的通道数必须等于 dim"

#         # 计算 Query
#         q1 = self.wq(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)
#         q2 = self.wq(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)
#         #print(q1.size())#torch.Size([4, 8, 64, 16])
#         if self.sr_ratio > 1:
#             x1_sr = self.sr(x1)
#             x1_sr = x1_sr.permute(0, 2, 3, 1)
#             x1_sr = self.norm(x1_sr)
#             x1_sr = x1_sr.permute(0, 3, 1, 2)
#             k1 = self.wk(x1_sr)
#             v1 = self.wv(x1_sr)
#         else:
#             k1 = self.wk(x1)
#             v1 = self.wv(x1)
#         k1 = k1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)
#         v1 = v1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)

#         if self.sr_ratio > 1:
#             x2_sr = self.sr(x2)
#             x2_sr = x2_sr.permute(0, 2, 3, 1)
#             x2_sr = self.norm(x2_sr)
#             x2_sr = x2_sr.permute(0, 3, 1, 2)
#             k2 = self.wk(x2_sr)
#             v2 = self.wv(x2_sr)
#         else:
#             k2 = self.wk(x2)
#             v2 = self.wv(x2)
#         k2 = k2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)
#         v2 = v2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)


#         # 计算注意力
#         attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
#         attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
#         attn1 = attn1.softmax(dim=-1)
#         attn1 = self.attn_drop(attn1)
#         attn2 = attn2.softmax(dim=-1)
#         attn2 = self.attn_drop(attn2)

# # # 创建一个与 attn1 维度相同的全 1 矩阵
# #         ones_matrix = torch.ones_like(attn1)

# # # 逐元素相减
# #         attn1 = ones_matrix - attn1
# #         attn2 = ones_matrix - attn2


#         # 应用注意力
#         x = torch.matmul(attn1, v2).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)
#         y = torch.matmul(attn2, v1).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)

#         # 合并输出
#         x = torch.cat([x, y], dim=1)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x


#     def forward(self, x1, x2):
#         B1, C1, H1, W1 = x1.shape
#         B2, C2, H2, W2 = x2.shape

#         # 确保输入特征的通道数与 dim 一致
#         assert C1 == self.wq.in_channels, "x1 的通道数必须等于 dim"
#         assert C2 == self.wk.in_channels, "x2 的通道数必须等于 dim"

#         # 计算 Query
#         q1 = self.wq(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)
#         q2 = self.wq(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)
#         #print(q1.size())#torch.Size([4, 8, 64, 16])
        # if self.sr_ratio > 1:
        #     x1_sr = self.sr1(x1)

        #     x1_sr = x1_sr.permute(0, 2, 3, 1)

        #     x1_sr = self.norm1(x1_sr)

        #     x1_sr = x1_sr.permute(0, 3, 1, 2)

        #     k1 = self.wk(x1_sr)

        #     v1 = self.wv(x1_sr)
        # else:
        #     k1 = self.wk(x1)

        #     v1 = self.wv(x1)
        # k1 = k1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)

        # v1 = v1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)


        # if self.sr_ratio > 1:
        #     x2_sr = self.sr1(x2)

        #     x2_sr = x2_sr.permute(0, 2, 3, 1)

        #     x2_sr = self.norm1(x2_sr)

        #     x2_sr = x2_sr.permute(0, 3, 1, 2)

        #     k2 = self.wk(x2_sr)

        #     v2 = self.wv(x2_sr)
        # else:
        #     k2 = self.wk(x2)

        #     v2 = self.wv(x2)
        # k2 = k2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)

        # v2 = v2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)



#         # 计算注意力
#         def chunk_attn(q, k, scale, chunk_size=32):
#             B, h, N, d = q.shape
#             M = k.shape[2]  # (H2//sr)*(W2//sr)
#             attn_weights = torch.zeros(B, h, N, M, device=q.device)
#             for i in range(0, N, chunk_size):
#                 q_chunk = q[:, :, i:i+chunk_size, :]
#                 attn_chunk = (q_chunk @ k.transpose(-2, -1)) * scale
#                 attn_chunk = attn_chunk.softmax(dim=-1)
#                 attn_weights[:, :, i:i+chunk_size, :] = attn_chunk
#             return attn_weights
#         attn1 = chunk_attn(q1, k2, self.scale)
#         attn2 = chunk_attn(q2, k1, self.scale)
#         # attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
#         # attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
#         # attn1 = attn1.softmax(dim=-1)
#         # attn1 = self.attn_drop(attn1)
#         # attn2 = attn2.softmax(dim=-1)
#         # attn2 = self.attn_drop(attn2)

#         # 应用注意力
#         # print("atten", attn1.shape)
#         # print("v2", v2.shape)
#         x = torch.matmul(attn1, v2).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)
#         y = torch.matmul(attn2, v1).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)

#         # 合并输出
#         x = torch.cat([x, y], dim=1)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

'''
交叉注意力＋spatial spatial CBAM
'''
#     def _upsample_(self, x, y):
#         _, _, H, W = y.size()
#         return F.interpolate(x, size=(H, W), mode='bilinear') 
    
#     def forward(self, pre_features, post_features):
        
#         # 选择设备
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # 将所有特征图移动到指定设备
#         pre_features = [feat.to(device) for feat in pre_features]
#         post_features = [feat.to(device) for feat in post_features]

#         pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

#         post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features
#         # print(pre_feat_4.size())# torch.Size([4, 768, 8, 8])
#         # print(pre_feat_3.size())#torch.Size([4, 384, 16, 16])
#         # print(pre_feat_2.size())# torch.Size([4, 192, 32, 32])
#         # print(pre_feat_1.size())# torch.Size([4, 96, 64, 64])

#         '''
#             Stage I 
#         '''
#         # print(pre_feat_4.size()) #torch.Size([4, 768, 8, 8])
#        # p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
#         #print(p41.size())  #torch.Size([4, 128, 8, 8])
#         B, C, H, W = pre_feat_4.size()
#         B, C, H, W = pre_feat_4.size()
#         pre_feat_4 =self.conv_768(pre_feat_4)
#         post_feat_4 = self.conv_768(post_feat_4)
#         p4 = self.crossattention4(pre_feat_4, pre_feat_4)
#         #print(p4.size())#torch.Size([4, 128, 8, 8])
#         '''
#             Stage II
#         '''
#         pre_feat_3 = self.conv_384(pre_feat_3)
#         post_feat_3 = self.conv_384(post_feat_3)
#         #print("pre 3:", pre_feat_3.size())#[4,128,16,16]
#         p3= self.crossattention3(pre_feat_3, post_feat_3) #[4,128,16,16]
#         #print("p3",p3.size())

#         p4 = self._upsample_(p4, p3)
#         #print(p4.size())
#         #print(p3.size())
#         p3 = self.fusion3(p4,p3)
#         #print("p33",p33.size())#([4, 128, 16, 16])     

#         '''
#             Stage III
#         '''
#         pre_feat_2 = self.conv_192(pre_feat_2)
#         post_feat_2 = self.conv_192(post_feat_2)
#         #print("pre 2:", pre_feat_2.size())#[4,128,32,32]
#         p2= self.crossattention2(pre_feat_2, post_feat_2)
#         #print(p2.size())

#         p3 = self._upsample_(p3, p2)
#         #print(p4.size())
#         #print(p3.size())
#         p2 = self.fusion3(p3,p2)
#         #print("p22",p22.size())#([8, 128, 32,32])     

#         '''
#             Stage Ⅳ
#         '''
#         pre_feat_1 = self.conv_96(pre_feat_1)
#         post_feat_1 = self.conv_96(post_feat_1)
#         #print("pre 1:", pre_feat_1.size())#[4,128,64,64]
#         p1= self.crossattention1(pre_feat_1, post_feat_1)
#         #print(p1.size())#[4,128,64,64]

#         p2 = self._upsample_(p2, p1)
#         #print(p4.size())
#         #print(p3.size())
#         p1 = self.fusion4(p2,p1)
#         #print("p11",p11.size())#[4,128,64,64]
#         return p1

'''
交叉注意力＋vss+ spatial spatial CBAM
'''
    # def _upsample_(self, x, y):
    #     _, _, H, W = y.size()
    #     return F.interpolate(x, size=(H, W), mode='bilinear') 
    
    # def forward(self, pre_features, post_features):
        
    #     # 选择设备
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     # 将所有特征图移动到指定设备
    #     pre_features = [feat.to(device) for feat in pre_features]
    #     post_features = [feat.to(device) for feat in post_features]

    #     pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

    #     post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features
    #     # print(pre_feat_4.size())# torch.Size([4, 768, 8, 8])
    #     # print(pre_feat_3.size())#torch.Size([4, 384, 16, 16])
    #     # print(pre_feat_2.size())# torch.Size([4, 192, 32, 32])
    #     # print(pre_feat_1.size())# torch.Size([4, 96, 64, 64])

    #     '''
    #         Stage I 
    #     '''
    #     # print(pre_feat_4.size()) #torch.Size([4, 768, 8, 8])
    #    # p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
    #     #print(p41.size())  #torch.Size([4, 128, 8, 8])
    #     B, C, H, W = pre_feat_4.size()
    #     B, C, H, W = pre_feat_4.size()
    #     pre_feat_4 =self.conv_768(pre_feat_4)
    #     post_feat_4 = self.conv_768(post_feat_4)
    #     p4 = self.crossattention(pre_feat_4, pre_feat_4)
    #     p4 = self.vssblock(p4)
    #     #print(p4.size())#torch.Size([4, 128, 8, 8])
    #     '''
    #         Stage II
    #     '''
    #     pre_feat_3 = self.conv_384(pre_feat_3)
    #     post_feat_3 = self.conv_384(post_feat_3)
    #     #print("pre 3:", pre_feat_3.size())#[4,128,16,16]
    #     p3= self.crossattention(pre_feat_3, post_feat_3) #[4,128,16,16]
    #     #print("p3",p3.size())

    #     p4 = self._upsample_(p4, p3)
    #     #print(p4.size())
    #     #print(p3.size())
    #     p3 = self.fusion3(p4,p3)
    #     #print("p33",p33.size())#([4, 128, 16, 16])     
    #     p3 = self.vssblock(p3)
    #     '''
    #         Stage III
    #     '''
    #     pre_feat_2 = self.conv_192(pre_feat_2)
    #     post_feat_2 = self.conv_192(post_feat_2)
    #     #print("pre 2:", pre_feat_2.size())#[4,128,32,32]
    #     p2= self.crossattention(pre_feat_2, post_feat_2)
    #     #print(p2.size())

    #     p3 = self._upsample_(p3, p2)
    #     #print(p4.size())
    #     #print(p3.size())
    #     p2 = self.fusion3(p3,p2)
    #     #print("p22",p22.size())#([8, 128, 32,32])     
    #     p2 = self.vssblock(p2)
    #     '''
    #         Stage Ⅳ
    #     '''
    #     pre_feat_1 = self.conv_96(pre_feat_1)
    #     post_feat_1 = self.conv_96(post_feat_1)
    #     #print("pre 1:", pre_feat_1.size())#[4,128,64,64]
    #     p1= self.crossattention(pre_feat_1, post_feat_1)
    #     #print(p1.size())#[4,128,64,64]

    #     p2 = self._upsample_(p2, p1)
    #     #print(p4.size())
    #     #print(p3.size())
    #     p1 = self.fusion4(p2,p1)
    #     #print("p11",p11.size())#[4,128,64,64]
    #     p1 = self.vssblock(p1)
    #     return p1
