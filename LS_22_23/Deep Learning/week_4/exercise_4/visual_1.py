# importing the required module
import matplotlib.pyplot as plt
rec1="1407/1407 [==============================] - 112s 79ms/step - loss: 1.8658 - accuracy: 0.2817 - val_loss: 1.5971 - val_accuracy: 0.3968" \
     "1407/1407 [==============================] - 112s 79ms/step - loss: 1.5221 - accuracy: 0.4241 - val_loss: 1.2995 - val_accuracy: 0.5138" \
     "1407/1407 [==============================] - 112s 80ms/step - loss: 1.3334 - accuracy: 0.5105 - val_loss: 1.1854 - val_accuracy: 0.5820" \
     "1407/1407 [==============================] - 112s 80ms/step - loss: 1.1983 - accuracy: 0.5672 - val_loss: 1.0820 - val_accuracy: 0.6276" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 1.0923 - accuracy: 0.6158 - val_loss: 0.8827 - val_accuracy: 0.6780" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.9947 - accuracy: 0.6520 - val_loss: 0.7751 - val_accuracy: 0.7284" \
     "1407/1407 [==============================] - 109s 78ms/step - loss: 0.9169 - accuracy: 0.6839 - val_loss: 0.9180 - val_accuracy: 0.6684" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.8532 - accuracy: 0.7088 - val_loss: 0.7262 - val_accuracy: 0.7472" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.7938 - accuracy: 0.7286 - val_loss: 0.6628 - val_accuracy: 0.7640" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.7433 - accuracy: 0.7502 - val_loss: 0.9703 - val_accuracy: 0.6788" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.6966 - accuracy: 0.7649 - val_loss: 0.6163 - val_accuracy: 0.7850" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.6591 - accuracy: 0.7795 - val_loss: 0.6180 - val_accuracy: 0.7848" \
     "1407/1407 [==============================] - 109s 78ms/step - loss: 0.6313 - accuracy: 0.7890 - val_loss: 0.6436 - val_accuracy: 0.7782" \
     "1407/1407 [==============================] - 109s 78ms/step - loss: 0.5969 - accuracy: 0.7991 - val_loss: 0.6593 - val_accuracy: 0.7730" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5703 - accuracy: 0.8088 - val_loss: 0.6458 - val_accuracy: 0.7858" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5572 - accuracy: 0.8121 - val_loss: 0.5439 - val_accuracy: 0.8120" \
     "1407/1407 [==============================] - 109s 78ms/step - loss: 0.5267 - accuracy: 0.8241 - val_loss: 0.6493 - val_accuracy: 0.7782" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5169 - accuracy: 0.8269 - val_loss: 0.5359 - val_accuracy: 0.8212" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5077 - accuracy: 0.8305 - val_loss: 0.5323 - val_accuracy: 0.8178" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4804 - accuracy: 0.8380 - val_loss: 0.5156 - val_accuracy: 0.8332" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4675 - accuracy: 0.8423 - val_loss: 0.5527 - val_accuracy: 0.8204" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4557 - accuracy: 0.8479 - val_loss: 0.4717 - val_accuracy: 0.8412" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4481 - accuracy: 0.8510 - val_loss: 0.4638 - val_accuracy: 0.8482" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4275 - accuracy: 0.8560 - val_loss: 0.6101 - val_accuracy: 0.8066" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4204 - accuracy: 0.8590 - val_loss: 0.5835 - val_accuracy: 0.8102" \
     "1407/1407 [==============================] - 116s 82ms/step - loss: 0.4158 - accuracy: 0.8604 - val_loss: 0.4889 - val_accuracy: 0.8378" \
     "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4080 - accuracy: 0.8640 - val_loss: 0.5206 - val_accuracy: 0.8378" \
     "1407/1407 [==============================] - 113s 80ms/step - loss: 0.4015 - accuracy: 0.8663 - val_loss: 0.4872 - val_accuracy: 0.8470" \
     "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3869 - accuracy: 0.8688 - val_loss: 0.4655 - val_accuracy: 0.8530" \
     "1407/1407 [==============================] - 112s 79ms/step - loss: 0.3814 - accuracy: 0.8698 - val_loss: 0.4700 - val_accuracy: 0.8502"

rec2="1407/1407 [==============================] - 112s 79ms/step - loss: 1.8658 - accuracy: 0.2817 - val_loss: 1.5971 - val_accuracy: 0.3968" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 1.5221 - accuracy: 0.4241 - val_loss: 1.2995 - val_accuracy: 0.5138" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 1.3334 - accuracy: 0.5105 - val_loss: 1.1854 - val_accuracy: 0.5820" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 1.1983 - accuracy: 0.5672 - val_loss: 1.0820 - val_accuracy: 0.6276" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 1.0923 - accuracy: 0.6158 - val_loss: 0.8827 - val_accuracy: 0.6780" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.9947 - accuracy: 0.6520 - val_loss: 0.7751 - val_accuracy: 0.7284" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.9169 - accuracy: 0.6839 - val_loss: 0.9180 - val_accuracy: 0.6684" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.8532 - accuracy: 0.7088 - val_loss: 0.7262 - val_accuracy: 0.7472" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.7938 - accuracy: 0.7286 - val_loss: 0.6628 - val_accuracy: 0.7640" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.7433 - accuracy: 0.7502 - val_loss: 0.9703 - val_accuracy: 0.6788" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6966 - accuracy: 0.7649 - val_loss: 0.6163 - val_accuracy: 0.7850" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.6591 - accuracy: 0.7795 - val_loss: 0.6180 - val_accuracy: 0.7848" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.6313 - accuracy: 0.7890 - val_loss: 0.6436 - val_accuracy: 0.7782" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5969 - accuracy: 0.7991 - val_loss: 0.6593 - val_accuracy: 0.7730" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5703 - accuracy: 0.8088 - val_loss: 0.6458 - val_accuracy: 0.7858" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5572 - accuracy: 0.8121 - val_loss: 0.5439 - val_accuracy: 0.8120" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5267 - accuracy: 0.8241 - val_loss: 0.6493 - val_accuracy: 0.7782" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5169 - accuracy: 0.8269 - val_loss: 0.5359 - val_accuracy: 0.8212" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5077 - accuracy: 0.8305 - val_loss: 0.5323 - val_accuracy: 0.8178" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4804 - accuracy: 0.8380 - val_loss: 0.5156 - val_accuracy: 0.8332" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4675 - accuracy: 0.8423 - val_loss: 0.5527 - val_accuracy: 0.8204" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4557 - accuracy: 0.8479 - val_loss: 0.4717 - val_accuracy: 0.8412" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4481 - accuracy: 0.8510 - val_loss: 0.4638 - val_accuracy: 0.8482" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4275 - accuracy: 0.8560 - val_loss: 0.6101 - val_accuracy: 0.8066" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4204 - accuracy: 0.8590 - val_loss: 0.5835 - val_accuracy: 0.8102" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4158 - accuracy: 0.8604 - val_loss: 0.4889 - val_accuracy: 0.8378" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4080 - accuracy: 0.8640 - val_loss: 0.5206 - val_accuracy: 0.8378" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.4015 - accuracy: 0.8663 - val_loss: 0.4872 - val_accuracy: 0.8470" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3869 - accuracy: 0.8688 - val_loss: 0.4655 - val_accuracy: 0.8530" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3814 - accuracy: 0.8698 - val_loss: 0.4700 - val_accuracy: 0.8502" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3802 - accuracy: 0.8734 - val_loss: 0.4955 - val_accuracy: 0.8432" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3717 - accuracy: 0.8741 - val_loss: 0.4772 - val_accuracy: 0.8438" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3604 - accuracy: 0.8795 - val_loss: 0.5609 - val_accuracy: 0.8262" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3564 - accuracy: 0.8787 - val_loss: 0.4824 - val_accuracy: 0.8468" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3575 - accuracy: 0.8780 - val_loss: 0.4750 - val_accuracy: 0.8486" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3484 - accuracy: 0.8825 - val_loss: 0.4556 - val_accuracy: 0.8518" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3451 - accuracy: 0.8823 - val_loss: 0.4786 - val_accuracy: 0.8512" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3362 - accuracy: 0.8850 - val_loss: 0.4751 - val_accuracy: 0.8538" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3318 - accuracy: 0.8875 - val_loss: 0.4955 - val_accuracy: 0.8520" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3292 - accuracy: 0.8887 - val_loss: 0.5331 - val_accuracy: 0.8400" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3286 - accuracy: 0.8887 - val_loss: 0.4656 - val_accuracy: 0.8544" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3175 - accuracy: 0.8912 - val_loss: 0.4557 - val_accuracy: 0.8582" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3126 - accuracy: 0.8930 - val_loss: 0.4910 - val_accuracy: 0.8596" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3151 - accuracy: 0.8928 - val_loss: 0.4912 - val_accuracy: 0.8520" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3066 - accuracy: 0.8958 - val_loss: 0.4572 - val_accuracy: 0.8586" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3027 - accuracy: 0.8960 - val_loss: 0.4362 - val_accuracy: 0.8674" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.3062 - accuracy: 0.8967 - val_loss: 0.5044 - val_accuracy: 0.8448" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2993 - accuracy: 0.8969 - val_loss: 0.4342 - val_accuracy: 0.8666" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2917 - accuracy: 0.9005 - val_loss: 0.5643 - val_accuracy: 0.8428" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2918 - accuracy: 0.9009 - val_loss: 0.4279 - val_accuracy: 0.8650" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2912 - accuracy: 0.9004 - val_loss: 0.4668 - val_accuracy: 0.8642" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2878 - accuracy: 0.9030 - val_loss: 0.4657 - val_accuracy: 0.8560" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2860 - accuracy: 0.9037 - val_loss: 0.4627 - val_accuracy: 0.8608" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2747 - accuracy: 0.9048 - val_loss: 0.4470 - val_accuracy: 0.8638" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2792 - accuracy: 0.9053 - val_loss: 0.4608 - val_accuracy: 0.8588" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2696 - accuracy: 0.9076 - val_loss: 0.4615 - val_accuracy: 0.8620" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2752 - accuracy: 0.9060 - val_loss: 0.4611 - val_accuracy: 0.8594" \
     "1407/1407 [==============================] - 110s 78ms/step - loss: 0.2655 - accuracy: 0.9119 - val_loss: 0.4619 - val_accuracy: 0.8564" \
     "1407/1407 [==============================] - 114s 81ms/step - loss: 0.2614 - accuracy: 0.9097 - val_loss: 0.4527 - val_accuracy: 0.8624" \
     "1407/1407 [==============================] - 114s 81ms/step - loss: 0.2687 - accuracy: 0.9072 - val_loss: 0.4318 - val_accuracy: 0.8678" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2642 - accuracy: 0.9111 - val_loss: 0.4679 - val_accuracy: 0.8662" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2551 - accuracy: 0.9124 - val_loss: 0.4852 - val_accuracy: 0.8582" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2549 - accuracy: 0.9129 - val_loss: 0.4945 - val_accuracy: 0.8512" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2528 - accuracy: 0.9137 - val_loss: 0.4636 - val_accuracy: 0.8660" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2500 - accuracy: 0.9141 - val_loss: 0.4911 - val_accuracy: 0.8654" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2507 - accuracy: 0.9148 - val_loss: 0.4782 - val_accuracy: 0.8688" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2504 - accuracy: 0.9144 - val_loss: 0.4816 - val_accuracy: 0.8600" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2497 - accuracy: 0.9158 - val_loss: 0.4942 - val_accuracy: 0.8642" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2541 - accuracy: 0.9139 - val_loss: 0.4757 - val_accuracy: 0.8596" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2483 - accuracy: 0.9162 - val_loss: 0.4653 - val_accuracy: 0.8630" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2437 - accuracy: 0.9173 - val_loss: 0.4581 - val_accuracy: 0.8638" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2373 - accuracy: 0.9188 - val_loss: 0.4920 - val_accuracy: 0.8598" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2394 - accuracy: 0.9184 - val_loss: 0.4779 - val_accuracy: 0.8650" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2350 - accuracy: 0.9195 - val_loss: 0.4967 - val_accuracy: 0.8558" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2387 - accuracy: 0.9193 - val_loss: 0.5131 - val_accuracy: 0.8608" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2320 - accuracy: 0.9204 - val_loss: 0.4781 - val_accuracy: 0.8664" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2327 - accuracy: 0.9217 - val_loss: 0.4834 - val_accuracy: 0.8630" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2288 - accuracy: 0.9229 - val_loss: 0.4497 - val_accuracy: 0.8700" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2325 - accuracy: 0.9205 - val_loss: 0.5124 - val_accuracy: 0.8566" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2265 - accuracy: 0.9238 - val_loss: 0.4828 - val_accuracy: 0.8590" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2235 - accuracy: 0.9242 - val_loss: 0.4942 - val_accuracy: 0.8704" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2276 - accuracy: 0.9215 - val_loss: 0.4696 - val_accuracy: 0.8706" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2264 - accuracy: 0.9245 - val_loss: 0.4950 - val_accuracy: 0.8582" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2168 - accuracy: 0.9260 - val_loss: 0.4947 - val_accuracy: 0.8606" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2159 - accuracy: 0.9268 - val_loss: 0.4894 - val_accuracy: 0.8672" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2188 - accuracy: 0.9257 - val_loss: 0.4975 - val_accuracy: 0.8614" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2221 - accuracy: 0.9234 - val_loss: 0.4611 - val_accuracy: 0.8714" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2126 - accuracy: 0.9291 - val_loss: 0.4699 - val_accuracy: 0.8678" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2194 - accuracy: 0.9273 - val_loss: 0.4677 - val_accuracy: 0.8714" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2092 - accuracy: 0.9298 - val_loss: 0.4356 - val_accuracy: 0.8766" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2120 - accuracy: 0.9283 - val_loss: 0.4630 - val_accuracy: 0.8694" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2152 - accuracy: 0.9263 - val_loss: 0.4731 - val_accuracy: 0.8682" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2158 - accuracy: 0.9273 - val_loss: 0.4694 - val_accuracy: 0.8656" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2047 - accuracy: 0.9293 - val_loss: 0.4782 - val_accuracy: 0.8694" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2074 - accuracy: 0.9306 - val_loss: 0.4832 - val_accuracy: 0.8690" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2108 - accuracy: 0.9306 - val_loss: 0.4617 - val_accuracy: 0.8708" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2031 - accuracy: 0.9324 - val_loss: 0.4989 - val_accuracy: 0.8590" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2037 - accuracy: 0.9304 - val_loss: 0.4780 - val_accuracy: 0.8688" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2052 - accuracy: 0.9295 - val_loss: 0.4952 - val_accuracy: 0.8654" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2021 - accuracy: 0.9323 - val_loss: 0.4881 - val_accuracy: 0.8678" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2038 - accuracy: 0.9313 - val_loss: 0.4638 - val_accuracy: 0.8716" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1999 - accuracy: 0.9320 - val_loss: 0.4784 - val_accuracy: 0.8688" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.2048 - accuracy: 0.9315 - val_loss: 0.4740 - val_accuracy: 0.8742" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1942 - accuracy: 0.9345 - val_loss: 0.4953 - val_accuracy: 0.8654" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1999 - accuracy: 0.9324 - val_loss: 0.4853 - val_accuracy: 0.8654" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1959 - accuracy: 0.9344 - val_loss: 0.4843 - val_accuracy: 0.8662" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1962 - accuracy: 0.9327 - val_loss: 0.4770 - val_accuracy: 0.8724" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1972 - accuracy: 0.9340 - val_loss: 0.5309 - val_accuracy: 0.8636" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1949 - accuracy: 0.9356 - val_loss: 0.4465 - val_accuracy: 0.8792" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1909 - accuracy: 0.9362 - val_loss: 0.4780 - val_accuracy: 0.8766" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1911 - accuracy: 0.9361 - val_loss: 0.5186 - val_accuracy: 0.8644" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1875 - accuracy: 0.9367 - val_loss: 0.4706 - val_accuracy: 0.8748" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1877 - accuracy: 0.9368 - val_loss: 0.4925 - val_accuracy: 0.8670" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1885 - accuracy: 0.9362 - val_loss: 0.5189 - val_accuracy: 0.8688" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1819 - accuracy: 0.9391 - val_loss: 0.5198 - val_accuracy: 0.8696" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1850 - accuracy: 0.9375 - val_loss: 0.4556 - val_accuracy: 0.8754" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1841 - accuracy: 0.9386 - val_loss: 0.4723 - val_accuracy: 0.8756" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1896 - accuracy: 0.9363 - val_loss: 0.4675 - val_accuracy: 0.8762" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1837 - accuracy: 0.9378 - val_loss: 0.5068 - val_accuracy: 0.8742" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1869 - accuracy: 0.9371 - val_loss: 0.5195 - val_accuracy: 0.8676" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1841 - accuracy: 0.9383 - val_loss: 0.4742 - val_accuracy: 0.8762" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1826 - accuracy: 0.9388 - val_loss: 0.4558 - val_accuracy: 0.8784" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1842 - accuracy: 0.9384 - val_loss: 0.4747 - val_accuracy: 0.8780" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1823 - accuracy: 0.9388 - val_loss: 0.4764 - val_accuracy: 0.8728" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1805 - accuracy: 0.9388 - val_loss: 0.4624 - val_accuracy: 0.8786" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1786 - accuracy: 0.9404 - val_loss: 0.5283 - val_accuracy: 0.8492" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1805 - accuracy: 0.9395 - val_loss: 0.4742 - val_accuracy: 0.8712" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1794 - accuracy: 0.9400 - val_loss: 0.5390 - val_accuracy: 0.8624" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1830 - accuracy: 0.9385 - val_loss: 0.4759 - val_accuracy: 0.8746" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1771 - accuracy: 0.9398 - val_loss: 0.5591 - val_accuracy: 0.8632" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1817 - accuracy: 0.9394 - val_loss: 0.4786 - val_accuracy: 0.8796" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1775 - accuracy: 0.9394 - val_loss: 0.5315 - val_accuracy: 0.8738" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1809 - accuracy: 0.9393 - val_loss: 0.4722 - val_accuracy: 0.8762" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1712 - accuracy: 0.9418 - val_loss: 0.5204 - val_accuracy: 0.8680" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1708 - accuracy: 0.9422 - val_loss: 0.4935 - val_accuracy: 0.8726" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1729 - accuracy: 0.9435 - val_loss: 0.4827 - val_accuracy: 0.8746" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1726 - accuracy: 0.9416 - val_loss: 0.5121 - val_accuracy: 0.8784" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1695 - accuracy: 0.9434 - val_loss: 0.5758 - val_accuracy: 0.8624" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1736 - accuracy: 0.9426 - val_loss: 0.4828 - val_accuracy: 0.8732" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1739 - accuracy: 0.9406 - val_loss: 0.5242 - val_accuracy: 0.8646" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1686 - accuracy: 0.9446 - val_loss: 0.5218 - val_accuracy: 0.8698" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1721 - accuracy: 0.9427 - val_loss: 0.4806 - val_accuracy: 0.8728" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1703 - accuracy: 0.9431 - val_loss: 0.5366 - val_accuracy: 0.8660" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1709 - accuracy: 0.9424 - val_loss: 0.4853 - val_accuracy: 0.8716" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1679 - accuracy: 0.9427 - val_loss: 0.5158 - val_accuracy: 0.8706" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1627 - accuracy: 0.9454 - val_loss: 0.4686 - val_accuracy: 0.8738" \
     "1407/1407 [==============================] - 112s 79ms/step - loss: 0.1652 - accuracy: 0.9444 - val_loss: 0.4811 - val_accuracy: 0.8716" \
     "1407/1407 [==============================] - 111s 79ms/step - loss: 0.1641 - accuracy: 0.9462 - val_loss: 0.4749 - val_accuracy: 0.8744" \
     "1407/1407 [==============================] - 112s 79ms/step - loss: 0.1643 - accuracy: 0.9449 - val_loss: 0.4675 - val_accuracy: 0.8794" \
     "1407/1407 [==============================] - 112s 79ms/step - loss: 0.1642 - accuracy: 0.9460 - val_loss: 0.4746 - val_accuracy: 0.8782"

l1 = "1407/1407 [==============================] - 140s 98ms/step - loss: 2.1069 - accuracy: 0.1680 - val_loss: 2.3985 - val_accuracy: 0.1450" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 1.9160 - accuracy: 0.2330 - val_loss: 1.7270 - val_accuracy: 0.3326" \
     "1407/1407 [==============================] - 161s 115ms/step - loss: 1.7679 - accuracy: 0.2994 - val_loss: 1.5226 - val_accuracy: 0.3886" \
     "1407/1407 [==============================] - 147s 104ms/step - loss: 1.6004 - accuracy: 0.3734 - val_loss: 1.4535 - val_accuracy: 0.4214" \
     "1407/1407 [==============================] - 159s 113ms/step - loss: 1.4228 - accuracy: 0.4370 - val_loss: 1.1317 - val_accuracy: 0.5882" \
     "1407/1407 [==============================] - 143s 102ms/step - loss: 1.2786 - accuracy: 0.5267 - val_loss: 0.9990 - val_accuracy: 0.6588" \
     "1407/1407 [==============================] - 142s 101ms/step - loss: 1.1457 - accuracy: 0.5908 - val_loss: 1.0674 - val_accuracy: 0.6552" \
     "1407/1407 [==============================] - 141s 100ms/step - loss: 1.0356 - accuracy: 0.6344 - val_loss: 0.9163 - val_accuracy: 0.6848" \
     "1407/1407 [==============================] - 141s 100ms/step - loss: 0.9330 - accuracy: 0.6796 - val_loss: 0.7504 - val_accuracy: 0.7432" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.8547 - accuracy: 0.7160 - val_loss: 0.6676 - val_accuracy: 0.7690" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.7952 - accuracy: 0.7404 - val_loss: 0.7520 - val_accuracy: 0.7468" \
     "1407/1407 [==============================] - 139s 98ms/step - loss: 0.7430 - accuracy: 0.7610 - val_loss: 0.6735 - val_accuracy: 0.7720" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.6982 - accuracy: 0.7761 - val_loss: 0.7599 - val_accuracy: 0.7634" \
     "1407/1407 [==============================] - 141s 100ms/step - loss: 0.6556 - accuracy: 0.7885 - val_loss: 0.6209 - val_accuracy: 0.7948" \
     "1407/1407 [==============================] - 139s 98ms/step - loss: 0.6304 - accuracy: 0.8017 - val_loss: 0.6866 - val_accuracy: 0.7826" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.5998 - accuracy: 0.8108 - val_loss: 0.6132 - val_accuracy: 0.7918" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.5581 - accuracy: 0.8251 - val_loss: 0.5724 - val_accuracy: 0.8202" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.5332 - accuracy: 0.8307 - val_loss: 0.5954 - val_accuracy: 0.8130" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.5058 - accuracy: 0.8397 - val_loss: 0.5728 - val_accuracy: 0.8248" \
     "1407/1407 [==============================] - 138s 98ms/step - loss: 0.4764 - accuracy: 0.8505 - val_loss: 0.5917 - val_accuracy: 0.8238" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.4556 - accuracy: 0.8545 - val_loss: 0.5542 - val_accuracy: 0.8286" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.4400 - accuracy: 0.8622 - val_loss: 0.6762 - val_accuracy: 0.8102" \
     "1407/1407 [==============================] - 140s 100ms/step - loss: 0.4179 - accuracy: 0.8683 - val_loss: 0.5053 - val_accuracy: 0.8410" \
     "1407/1407 [==============================] - 143s 102ms/step - loss: 0.4067 - accuracy: 0.8711 - val_loss: 0.6060 - val_accuracy: 0.8236" \
     "1407/1407 [==============================] - 140s 100ms/step - loss: 0.3927 - accuracy: 0.8758 - val_loss: 0.5332 - val_accuracy: 0.8394" \
     "1407/1407 [==============================] - 140s 100ms/step - loss: 0.3780 - accuracy: 0.8796 - val_loss: 0.6249 - val_accuracy: 0.8132" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.3583 - accuracy: 0.8858 - val_loss: 0.5660 - val_accuracy: 0.8388" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.3473 - accuracy: 0.8892 - val_loss: 0.5376 - val_accuracy: 0.8448" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.3338 - accuracy: 0.8942 - val_loss: 0.6639 - val_accuracy: 0.8174" \
     "1407/1407 [==============================] - 139s 99ms/step - loss: 0.3266 - accuracy: 0.8964 - val_loss: 0.6214 - val_accuracy: 0.8346"

rec3 = "1407/1407 [==============================] - 113s 79ms/step - loss: 1.9885 - accuracy: 0.2402 - val_loss: 2.1166 - val_accuracy: 0.2574" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 1.7466 - accuracy: 0.3217 - val_loss: 1.5810 - val_accuracy: 0.4014" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 1.6089 - accuracy: 0.3819 - val_loss: 1.3817 - val_accuracy: 0.4960" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 1.5044 - accuracy: 0.4408 - val_loss: 1.6337 - val_accuracy: 0.4772" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 1.3959 - accuracy: 0.4899 - val_loss: 1.3207 - val_accuracy: 0.5776" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 1.2812 - accuracy: 0.5428 - val_loss: 1.3200 - val_accuracy: 0.5292" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 1.1946 - accuracy: 0.5786 - val_loss: 1.0035 - val_accuracy: 0.6400" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 1.1204 - accuracy: 0.6098 - val_loss: 0.9906 - val_accuracy: 0.6454" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 1.0672 - accuracy: 0.6320 - val_loss: 1.0424 - val_accuracy: 0.6462" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.9972 - accuracy: 0.6603 - val_loss: 0.8702 - val_accuracy: 0.6900" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.9372 - accuracy: 0.6853 - val_loss: 0.6970 - val_accuracy: 0.7508" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.8925 - accuracy: 0.7006 - val_loss: 0.8138 - val_accuracy: 0.7258" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.8492 - accuracy: 0.7152 - val_loss: 0.7115 - val_accuracy: 0.7498" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.8132 - accuracy: 0.7271 - val_loss: 0.6720 - val_accuracy: 0.7620" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.7923 - accuracy: 0.7348 - val_loss: 0.8691 - val_accuracy: 0.7258" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.7687 - accuracy: 0.7448 - val_loss: 0.8327 - val_accuracy: 0.7324" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.7372 - accuracy: 0.7525 - val_loss: 0.6744 - val_accuracy: 0.7732" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.7266 - accuracy: 0.7588 - val_loss: 0.7036 - val_accuracy: 0.7686" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.7112 - accuracy: 0.7628 - val_loss: 0.5652 - val_accuracy: 0.8038" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6952 - accuracy: 0.7708 - val_loss: 0.6508 - val_accuracy: 0.7856" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6846 - accuracy: 0.7732 - val_loss: 0.6779 - val_accuracy: 0.7704" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6751 - accuracy: 0.7787 - val_loss: 0.6314 - val_accuracy: 0.7830" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6557 - accuracy: 0.7815 - val_loss: 0.8926 - val_accuracy: 0.7262" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6423 - accuracy: 0.7861 - val_loss: 0.7573 - val_accuracy: 0.7692" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6414 - accuracy: 0.7876 - val_loss: 0.5434 - val_accuracy: 0.8184" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6346 - accuracy: 0.7908 - val_loss: 0.5401 - val_accuracy: 0.8180" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6255 - accuracy: 0.7928 - val_loss: 0.6038 - val_accuracy: 0.8054" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6191 - accuracy: 0.7944 - val_loss: 0.6628 - val_accuracy: 0.7900" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6133 - accuracy: 0.7939 - val_loss: 0.5561 - val_accuracy: 0.8160" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.6127 - accuracy: 0.7957 - val_loss: 0.6112 - val_accuracy: 0.8034" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5977 - accuracy: 0.8020 - val_loss: 0.5660 - val_accuracy: 0.8170" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5936 - accuracy: 0.8029 - val_loss: 0.5644 - val_accuracy: 0.8188" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5828 - accuracy: 0.8097 - val_loss: 0.5232 - val_accuracy: 0.8226" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5791 - accuracy: 0.8079 - val_loss: 0.5335 - val_accuracy: 0.8234" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5702 - accuracy: 0.8102 - val_loss: 0.6378 - val_accuracy: 0.8068" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5716 - accuracy: 0.8105 - val_loss: 0.5740 - val_accuracy: 0.8186" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5670 - accuracy: 0.8126 - val_loss: 0.5109 - val_accuracy: 0.8346" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5567 - accuracy: 0.8148 - val_loss: 0.5518 - val_accuracy: 0.8210" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5571 - accuracy: 0.8145 - val_loss: 0.5269 - val_accuracy: 0.8312" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.5547 - accuracy: 0.8159 - val_loss: 0.5127 - val_accuracy: 0.8352" \
       "1407/1407 [==============================] - 110s 78ms/step - loss: 0.5476 - accuracy: 0.8180 - val_loss: 0.6079 - val_accuracy: 0.8082" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5457 - accuracy: 0.8175 - val_loss: 0.4788 - val_accuracy: 0.8362" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5441 - accuracy: 0.8186 - val_loss: 0.6038 - val_accuracy: 0.8120" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5366 - accuracy: 0.8190 - val_loss: 0.4668 - val_accuracy: 0.8462" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5353 - accuracy: 0.8208 - val_loss: 0.4942 - val_accuracy: 0.8370" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5320 - accuracy: 0.8206 - val_loss: 0.5059 - val_accuracy: 0.8364" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5320 - accuracy: 0.8216 - val_loss: 0.5797 - val_accuracy: 0.8270" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5268 - accuracy: 0.8253 - val_loss: 0.4687 - val_accuracy: 0.8450" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5262 - accuracy: 0.8256 - val_loss: 0.5677 - val_accuracy: 0.8212" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5222 - accuracy: 0.8242 - val_loss: 0.5172 - val_accuracy: 0.8372" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.5190 - accuracy: 0.8278 - val_loss: 0.4898 - val_accuracy: 0.8442" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5078 - accuracy: 0.8305 - val_loss: 0.4733 - val_accuracy: 0.8460" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5067 - accuracy: 0.8323 - val_loss: 0.4409 - val_accuracy: 0.8504" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5034 - accuracy: 0.8324 - val_loss: 0.4276 - val_accuracy: 0.8600" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.5061 - accuracy: 0.8322 - val_loss: 0.4548 - val_accuracy: 0.8510" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4992 - accuracy: 0.8326 - val_loss: 0.4251 - val_accuracy: 0.8566" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.5016 - accuracy: 0.8313 - val_loss: 0.4734 - val_accuracy: 0.8488" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4940 - accuracy: 0.8353 - val_loss: 0.4770 - val_accuracy: 0.8510" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4926 - accuracy: 0.8355 - val_loss: 0.5368 - val_accuracy: 0.8300" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4924 - accuracy: 0.8366 - val_loss: 0.5062 - val_accuracy: 0.8322" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4874 - accuracy: 0.8361 - val_loss: 0.4672 - val_accuracy: 0.8540" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4841 - accuracy: 0.8397 - val_loss: 0.4420 - val_accuracy: 0.8570" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4858 - accuracy: 0.8395 - val_loss: 0.5939 - val_accuracy: 0.8208" \
       "1407/1407 [==============================] - 118s 84ms/step - loss: 0.4830 - accuracy: 0.8391 - val_loss: 0.4415 - val_accuracy: 0.8560" \
       "1407/1407 [==============================] - 120s 85ms/step - loss: 0.4821 - accuracy: 0.8390 - val_loss: 0.4550 - val_accuracy: 0.8542" \
       "1407/1407 [==============================] - 116s 83ms/step - loss: 0.4869 - accuracy: 0.8375 - val_loss: 0.4733 - val_accuracy: 0.8482" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.4834 - accuracy: 0.8386 - val_loss: 0.5351 - val_accuracy: 0.8392" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4742 - accuracy: 0.8416 - val_loss: 0.4702 - val_accuracy: 0.8526" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4770 - accuracy: 0.8417 - val_loss: 0.4636 - val_accuracy: 0.8510" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4768 - accuracy: 0.8412 - val_loss: 0.5110 - val_accuracy: 0.8440" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4678 - accuracy: 0.8438 - val_loss: 0.4973 - val_accuracy: 0.8460" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4640 - accuracy: 0.8460 - val_loss: 0.4677 - val_accuracy: 0.8568" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4670 - accuracy: 0.8446 - val_loss: 0.4925 - val_accuracy: 0.8456" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4702 - accuracy: 0.8404 - val_loss: 0.4739 - val_accuracy: 0.8492" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4639 - accuracy: 0.8453 - val_loss: 0.4745 - val_accuracy: 0.8446" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4597 - accuracy: 0.8478 - val_loss: 0.4809 - val_accuracy: 0.8470" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4660 - accuracy: 0.8442 - val_loss: 0.4096 - val_accuracy: 0.8692" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4624 - accuracy: 0.8441 - val_loss: 0.4174 - val_accuracy: 0.8628" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4571 - accuracy: 0.8474 - val_loss: 0.4572 - val_accuracy: 0.8486" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4549 - accuracy: 0.8487 - val_loss: 0.4499 - val_accuracy: 0.8558" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4582 - accuracy: 0.8464 - val_loss: 0.4411 - val_accuracy: 0.8564" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4562 - accuracy: 0.8506 - val_loss: 0.4515 - val_accuracy: 0.8564" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4554 - accuracy: 0.8483 - val_loss: 0.4910 - val_accuracy: 0.8432" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4461 - accuracy: 0.8500 - val_loss: 0.4900 - val_accuracy: 0.8542" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4471 - accuracy: 0.8515 - val_loss: 0.4593 - val_accuracy: 0.8556" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4485 - accuracy: 0.8492 - val_loss: 0.3990 - val_accuracy: 0.8710" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4477 - accuracy: 0.8516 - val_loss: 0.4228 - val_accuracy: 0.8644" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4434 - accuracy: 0.8522 - val_loss: 0.4160 - val_accuracy: 0.8664" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4467 - accuracy: 0.8520 - val_loss: 0.5583 - val_accuracy: 0.8300" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4391 - accuracy: 0.8554 - val_loss: 0.3936 - val_accuracy: 0.8686" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4427 - accuracy: 0.8528 - val_loss: 0.4489 - val_accuracy: 0.8568" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4441 - accuracy: 0.8531 - val_loss: 0.4334 - val_accuracy: 0.8640" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4369 - accuracy: 0.8545 - val_loss: 0.4142 - val_accuracy: 0.8676" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4349 - accuracy: 0.8558 - val_loss: 0.4943 - val_accuracy: 0.8484" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4360 - accuracy: 0.8532 - val_loss: 0.5034 - val_accuracy: 0.8468" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4316 - accuracy: 0.8577 - val_loss: 0.4502 - val_accuracy: 0.8574" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4373 - accuracy: 0.8546 - val_loss: 0.4734 - val_accuracy: 0.8548" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4310 - accuracy: 0.8546 - val_loss: 0.4261 - val_accuracy: 0.8646" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4283 - accuracy: 0.8565 - val_loss: 0.4553 - val_accuracy: 0.8520" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4334 - accuracy: 0.8564 - val_loss: 0.3912 - val_accuracy: 0.8716" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4307 - accuracy: 0.8560 - val_loss: 0.3919 - val_accuracy: 0.8742" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4310 - accuracy: 0.8567 - val_loss: 0.4659 - val_accuracy: 0.8580" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4351 - accuracy: 0.8542 - val_loss: 0.4434 - val_accuracy: 0.8594" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4243 - accuracy: 0.8592 - val_loss: 0.4358 - val_accuracy: 0.8598" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4249 - accuracy: 0.8566 - val_loss: 0.4804 - val_accuracy: 0.8472" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4200 - accuracy: 0.8594 - val_loss: 0.4283 - val_accuracy: 0.8628" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4248 - accuracy: 0.8567 - val_loss: 0.4323 - val_accuracy: 0.8664" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4263 - accuracy: 0.8570 - val_loss: 0.4023 - val_accuracy: 0.8666" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4193 - accuracy: 0.8582 - val_loss: 0.3984 - val_accuracy: 0.8730" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4203 - accuracy: 0.8590 - val_loss: 0.4798 - val_accuracy: 0.8574" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4174 - accuracy: 0.8602 - val_loss: 0.4279 - val_accuracy: 0.8616" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4184 - accuracy: 0.8614 - val_loss: 0.4545 - val_accuracy: 0.8588" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4217 - accuracy: 0.8592 - val_loss: 0.3936 - val_accuracy: 0.8762" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4179 - accuracy: 0.8602 - val_loss: 0.4335 - val_accuracy: 0.8678" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4191 - accuracy: 0.8594 - val_loss: 0.4408 - val_accuracy: 0.8610" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4213 - accuracy: 0.8597 - val_loss: 0.4045 - val_accuracy: 0.8762" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4210 - accuracy: 0.8604 - val_loss: 0.3780 - val_accuracy: 0.8734" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4178 - accuracy: 0.8593 - val_loss: 0.4665 - val_accuracy: 0.8562" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4161 - accuracy: 0.8602 - val_loss: 0.3716 - val_accuracy: 0.8788" \
       "1407/1407 [==============================] - 111s 79ms/step - loss: 0.4204 - accuracy: 0.8593 - val_loss: 0.4075 - val_accuracy: 0.8724" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4094 - accuracy: 0.8646 - val_loss: 0.4031 - val_accuracy: 0.8724" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4065 - accuracy: 0.8658 - val_loss: 0.3779 - val_accuracy: 0.8778" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4064 - accuracy: 0.8639 - val_loss: 0.4834 - val_accuracy: 0.8498" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4099 - accuracy: 0.8618 - val_loss: 0.3711 - val_accuracy: 0.8758" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4118 - accuracy: 0.8648 - val_loss: 0.3885 - val_accuracy: 0.8776" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4104 - accuracy: 0.8632 - val_loss: 0.4870 - val_accuracy: 0.8480" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4123 - accuracy: 0.8629 - val_loss: 0.3905 - val_accuracy: 0.8692" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4042 - accuracy: 0.8640 - val_loss: 0.3825 - val_accuracy: 0.8788" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4104 - accuracy: 0.8640 - val_loss: 0.4148 - val_accuracy: 0.8672" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4072 - accuracy: 0.8632 - val_loss: 0.3558 - val_accuracy: 0.8848" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4044 - accuracy: 0.8658 - val_loss: 0.4283 - val_accuracy: 0.8656" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4107 - accuracy: 0.8624 - val_loss: 0.4577 - val_accuracy: 0.8632" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4002 - accuracy: 0.8662 - val_loss: 0.3882 - val_accuracy: 0.8754" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4006 - accuracy: 0.8675 - val_loss: 0.3895 - val_accuracy: 0.8768" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.4085 - accuracy: 0.8628 - val_loss: 0.4406 - val_accuracy: 0.8658" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4004 - accuracy: 0.8656 - val_loss: 0.3327 - val_accuracy: 0.8904" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.3988 - accuracy: 0.8667 - val_loss: 0.4397 - val_accuracy: 0.8636" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4072 - accuracy: 0.8647 - val_loss: 0.3749 - val_accuracy: 0.8848" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4007 - accuracy: 0.8678 - val_loss: 0.5102 - val_accuracy: 0.8506" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4006 - accuracy: 0.8656 - val_loss: 0.4131 - val_accuracy: 0.8746" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.4009 - accuracy: 0.8652 - val_loss: 0.4123 - val_accuracy: 0.8660" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3980 - accuracy: 0.8665 - val_loss: 0.3750 - val_accuracy: 0.8758" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.3927 - accuracy: 0.8679 - val_loss: 0.4169 - val_accuracy: 0.8718" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.3942 - accuracy: 0.8690 - val_loss: 0.4236 - val_accuracy: 0.8688" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.3974 - accuracy: 0.8677 - val_loss: 0.4162 - val_accuracy: 0.8684" \
       "1407/1407 [==============================] - 112s 79ms/step - loss: 0.3941 - accuracy: 0.8681 - val_loss: 0.3614 - val_accuracy: 0.8866" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3922 - accuracy: 0.8677 - val_loss: 0.3942 - val_accuracy: 0.8806" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3963 - accuracy: 0.8685 - val_loss: 0.4121 - val_accuracy: 0.8744" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3918 - accuracy: 0.8694 - val_loss: 0.4130 - val_accuracy: 0.8698" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3913 - accuracy: 0.8683 - val_loss: 0.4698 - val_accuracy: 0.8606" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3988 - accuracy: 0.8686 - val_loss: 0.3876 - val_accuracy: 0.8770" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3965 - accuracy: 0.8675 - val_loss: 0.4051 - val_accuracy: 0.8714" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3884 - accuracy: 0.8708 - val_loss: 0.3846 - val_accuracy: 0.8802" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3919 - accuracy: 0.8684 - val_loss: 0.3594 - val_accuracy: 0.8868" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3932 - accuracy: 0.8688 - val_loss: 0.4317 - val_accuracy: 0.8700" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3930 - accuracy: 0.8701 - val_loss: 0.3752 - val_accuracy: 0.8810" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3880 - accuracy: 0.8718 - val_loss: 0.3429 - val_accuracy: 0.8906" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3866 - accuracy: 0.8709 - val_loss: 0.3584 - val_accuracy: 0.8846" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3872 - accuracy: 0.8703 - val_loss: 0.3591 - val_accuracy: 0.8824" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3874 - accuracy: 0.8724 - val_loss: 0.4744 - val_accuracy: 0.8568" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3882 - accuracy: 0.8682 - val_loss: 0.4054 - val_accuracy: 0.8704" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3813 - accuracy: 0.8719 - val_loss: 0.3545 - val_accuracy: 0.8846" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3913 - accuracy: 0.8700 - val_loss: 0.4679 - val_accuracy: 0.8602" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3861 - accuracy: 0.8695 - val_loss: 0.3551 - val_accuracy: 0.8790" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3818 - accuracy: 0.8710 - val_loss: 0.3637 - val_accuracy: 0.8886" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3835 - accuracy: 0.8698 - val_loss: 0.3775 - val_accuracy: 0.8852" \
       "1407/1407 [==============================] - 112s 80ms/step - loss: 0.3801 - accuracy: 0.8731 - val_loss: 0.3527 - val_accuracy: 0.8842" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3868 - accuracy: 0.8704 - val_loss: 0.4227 - val_accuracy: 0.8756" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3805 - accuracy: 0.8753 - val_loss: 0.3776 - val_accuracy: 0.8756" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3853 - accuracy: 0.8687 - val_loss: 0.4762 - val_accuracy: 0.8628" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3779 - accuracy: 0.8742 - val_loss: 0.4244 - val_accuracy: 0.8784" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3860 - accuracy: 0.8692 - val_loss: 0.4694 - val_accuracy: 0.8608" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3844 - accuracy: 0.8719 - val_loss: 0.3939 - val_accuracy: 0.8746" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3777 - accuracy: 0.8732 - val_loss: 0.3733 - val_accuracy: 0.8824" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3865 - accuracy: 0.8709 - val_loss: 0.3538 - val_accuracy: 0.8876" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3826 - accuracy: 0.8746 - val_loss: 0.4345 - val_accuracy: 0.8718" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3759 - accuracy: 0.8732 - val_loss: 0.3971 - val_accuracy: 0.8750" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3785 - accuracy: 0.8747 - val_loss: 0.4182 - val_accuracy: 0.8684" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3745 - accuracy: 0.8760 - val_loss: 0.4347 - val_accuracy: 0.8682" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3793 - accuracy: 0.8726 - val_loss: 0.4473 - val_accuracy: 0.8588" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3798 - accuracy: 0.8739 - val_loss: 0.4160 - val_accuracy: 0.8718" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3712 - accuracy: 0.8759 - val_loss: 0.4330 - val_accuracy: 0.8678" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3764 - accuracy: 0.8755 - val_loss: 0.3818 - val_accuracy: 0.8856" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3819 - accuracy: 0.8728 - val_loss: 0.3994 - val_accuracy: 0.8784" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3732 - accuracy: 0.8745 - val_loss: 0.3660 - val_accuracy: 0.8854" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3771 - accuracy: 0.8730 - val_loss: 0.3789 - val_accuracy: 0.8806" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3727 - accuracy: 0.8761 - val_loss: 0.4139 - val_accuracy: 0.8766" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3739 - accuracy: 0.8763 - val_loss: 0.3783 - val_accuracy: 0.8848" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3723 - accuracy: 0.8748 - val_loss: 0.4028 - val_accuracy: 0.8772" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3736 - accuracy: 0.8767 - val_loss: 0.3708 - val_accuracy: 0.8834" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3782 - accuracy: 0.8737 - val_loss: 0.3822 - val_accuracy: 0.8758" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3695 - accuracy: 0.8780 - val_loss: 0.4827 - val_accuracy: 0.8540" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3683 - accuracy: 0.8776 - val_loss: 0.4182 - val_accuracy: 0.8756" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3709 - accuracy: 0.8758 - val_loss: 0.4072 - val_accuracy: 0.8764" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3752 - accuracy: 0.8747 - val_loss: 0.3849 - val_accuracy: 0.8842" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3672 - accuracy: 0.8763 - val_loss: 0.3552 - val_accuracy: 0.8906" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3690 - accuracy: 0.8770 - val_loss: 0.3841 - val_accuracy: 0.8748" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3699 - accuracy: 0.8763 - val_loss: 0.4332 - val_accuracy: 0.8644" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3668 - accuracy: 0.8775 - val_loss: 0.4929 - val_accuracy: 0.8548" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3689 - accuracy: 0.8762 - val_loss: 0.3743 - val_accuracy: 0.8854" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3701 - accuracy: 0.8785 - val_loss: 0.3913 - val_accuracy: 0.8826" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3673 - accuracy: 0.8761 - val_loss: 0.3824 - val_accuracy: 0.8788" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3700 - accuracy: 0.8751 - val_loss: 0.3957 - val_accuracy: 0.8776" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3707 - accuracy: 0.8748 - val_loss: 0.5057 - val_accuracy: 0.8536" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3667 - accuracy: 0.8774 - val_loss: 0.3914 - val_accuracy: 0.8788" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3644 - accuracy: 0.8784 - val_loss: 0.3627 - val_accuracy: 0.8914" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3633 - accuracy: 0.8776 - val_loss: 0.3568 - val_accuracy: 0.8898" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3688 - accuracy: 0.8777 - val_loss: 0.3887 - val_accuracy: 0.8788" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3660 - accuracy: 0.8784 - val_loss: 0.3897 - val_accuracy: 0.8846" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3711 - accuracy: 0.8769 - val_loss: 0.3892 - val_accuracy: 0.8870" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3666 - accuracy: 0.8781 - val_loss: 0.3670 - val_accuracy: 0.8858" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3667 - accuracy: 0.8770 - val_loss: 0.4381 - val_accuracy: 0.8696" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3610 - accuracy: 0.8790 - val_loss: 0.3945 - val_accuracy: 0.8828" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3634 - accuracy: 0.8794 - val_loss: 0.3649 - val_accuracy: 0.8896" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3615 - accuracy: 0.8795 - val_loss: 0.3611 - val_accuracy: 0.8872" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3651 - accuracy: 0.8773 - val_loss: 0.3551 - val_accuracy: 0.8932" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3669 - accuracy: 0.8791 - val_loss: 0.3565 - val_accuracy: 0.8910" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3627 - accuracy: 0.8802 - val_loss: 0.4362 - val_accuracy: 0.8712" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3633 - accuracy: 0.8776 - val_loss: 0.3651 - val_accuracy: 0.8872" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3663 - accuracy: 0.8778 - val_loss: 0.3586 - val_accuracy: 0.8844" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3658 - accuracy: 0.8764 - val_loss: 0.3870 - val_accuracy: 0.8840" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3642 - accuracy: 0.8789 - val_loss: 0.4111 - val_accuracy: 0.8762" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3631 - accuracy: 0.8787 - val_loss: 0.3805 - val_accuracy: 0.8888" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3592 - accuracy: 0.8809 - val_loss: 0.3775 - val_accuracy: 0.8866" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3598 - accuracy: 0.8810 - val_loss: 0.4207 - val_accuracy: 0.8720" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3578 - accuracy: 0.8817 - val_loss: 0.3922 - val_accuracy: 0.8796" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3644 - accuracy: 0.8784 - val_loss: 0.3537 - val_accuracy: 0.8924" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3621 - accuracy: 0.8803 - val_loss: 0.4222 - val_accuracy: 0.8720" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3619 - accuracy: 0.8783 - val_loss: 0.3474 - val_accuracy: 0.8918" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3560 - accuracy: 0.8813 - val_loss: 0.4272 - val_accuracy: 0.8726" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3589 - accuracy: 0.8816 - val_loss: 0.4527 - val_accuracy: 0.8710" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3596 - accuracy: 0.8794 - val_loss: 0.3891 - val_accuracy: 0.8778" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3576 - accuracy: 0.8808 - val_loss: 0.3863 - val_accuracy: 0.8834" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3566 - accuracy: 0.8815 - val_loss: 0.3688 - val_accuracy: 0.8864" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3591 - accuracy: 0.8805 - val_loss: 0.3879 - val_accuracy: 0.8830" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3570 - accuracy: 0.8784 - val_loss: 0.3726 - val_accuracy: 0.8810" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3561 - accuracy: 0.8820 - val_loss: 0.3903 - val_accuracy: 0.8784" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3570 - accuracy: 0.8810 - val_loss: 0.3526 - val_accuracy: 0.8818" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3596 - accuracy: 0.8795 - val_loss: 0.3627 - val_accuracy: 0.8862" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3526 - accuracy: 0.8811 - val_loss: 0.3719 - val_accuracy: 0.8828" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3516 - accuracy: 0.8825 - val_loss: 0.3719 - val_accuracy: 0.8826" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3557 - accuracy: 0.8818 - val_loss: 0.3638 - val_accuracy: 0.8862" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3609 - accuracy: 0.8811 - val_loss: 0.3685 - val_accuracy: 0.8874" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3530 - accuracy: 0.8829 - val_loss: 0.4695 - val_accuracy: 0.8680" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3610 - accuracy: 0.8792 - val_loss: 0.3985 - val_accuracy: 0.8762" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3547 - accuracy: 0.8800 - val_loss: 0.3361 - val_accuracy: 0.8940" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3555 - accuracy: 0.8823 - val_loss: 0.3680 - val_accuracy: 0.8858" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3573 - accuracy: 0.8804 - val_loss: 0.3322 - val_accuracy: 0.8954" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3530 - accuracy: 0.8824 - val_loss: 0.3778 - val_accuracy: 0.8874" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3527 - accuracy: 0.8813 - val_loss: 0.3503 - val_accuracy: 0.8920" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3542 - accuracy: 0.8808 - val_loss: 0.3362 - val_accuracy: 0.8938" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3482 - accuracy: 0.8824 - val_loss: 0.4061 - val_accuracy: 0.8796" \
       "1407/1407 [==============================] - 115s 82ms/step - loss: 0.3546 - accuracy: 0.8826 - val_loss: 0.3481 - val_accuracy: 0.8942" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3526 - accuracy: 0.8817 - val_loss: 0.3990 - val_accuracy: 0.8754" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3500 - accuracy: 0.8838 - val_loss: 0.4049 - val_accuracy: 0.8780" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3527 - accuracy: 0.8827 - val_loss: 0.3847 - val_accuracy: 0.8818" \
       "1407/1407 [==============================] - 113s 80ms/step - loss: 0.3549 - accuracy: 0.8816 - val_loss: 0.4359 - val_accuracy: 0.8736" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3458 - accuracy: 0.8850 - val_loss: 0.4597 - val_accuracy: 0.8676" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3524 - accuracy: 0.8834 - val_loss: 0.4143 - val_accuracy: 0.8726" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3476 - accuracy: 0.8844 - val_loss: 0.4626 - val_accuracy: 0.8590" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3510 - accuracy: 0.8839 - val_loss: 0.3733 - val_accuracy: 0.8874" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3517 - accuracy: 0.8812 - val_loss: 0.3999 - val_accuracy: 0.8770" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3432 - accuracy: 0.8843 - val_loss: 0.3850 - val_accuracy: 0.8846" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3451 - accuracy: 0.8834 - val_loss: 0.4134 - val_accuracy: 0.8762" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3469 - accuracy: 0.8839 - val_loss: 0.4037 - val_accuracy: 0.8804" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3487 - accuracy: 0.8837 - val_loss: 0.3298 - val_accuracy: 0.8986" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3476 - accuracy: 0.8847 - val_loss: 0.3624 - val_accuracy: 0.8900" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3479 - accuracy: 0.8840 - val_loss: 0.3638 - val_accuracy: 0.8852" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3518 - accuracy: 0.8809 - val_loss: 0.3820 - val_accuracy: 0.8862" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3461 - accuracy: 0.8856 - val_loss: 0.3255 - val_accuracy: 0.8910" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3464 - accuracy: 0.8848 - val_loss: 0.3881 - val_accuracy: 0.8784" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3493 - accuracy: 0.8844 - val_loss: 0.3895 - val_accuracy: 0.8784" \
       "1407/1407 [==============================] - 113s 81ms/step - loss: 0.3451 - accuracy: 0.8833 - val_loss: 0.3921 - val_accuracy: 0.8862" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3500 - accuracy: 0.8831 - val_loss: 0.3751 - val_accuracy: 0.8872" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3424 - accuracy: 0.8855 - val_loss: 0.3644 - val_accuracy: 0.8874" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3547 - accuracy: 0.8822 - val_loss: 0.3631 - val_accuracy: 0.8916" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3458 - accuracy: 0.8855 - val_loss: 0.3881 - val_accuracy: 0.8862" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3502 - accuracy: 0.8834 - val_loss: 0.3556 - val_accuracy: 0.8876" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3490 - accuracy: 0.8844 - val_loss: 0.4285 - val_accuracy: 0.8726" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3497 - accuracy: 0.8830 - val_loss: 0.3555 - val_accuracy: 0.8880" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3502 - accuracy: 0.8826 - val_loss: 0.3775 - val_accuracy: 0.8806" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3499 - accuracy: 0.8826 - val_loss: 0.3712 - val_accuracy: 0.8866" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3430 - accuracy: 0.8855 - val_loss: 0.3963 - val_accuracy: 0.8818" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3461 - accuracy: 0.8841 - val_loss: 0.3574 - val_accuracy: 0.8886" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3411 - accuracy: 0.8855 - val_loss: 0.3652 - val_accuracy: 0.8864" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3477 - accuracy: 0.8829 - val_loss: 0.3448 - val_accuracy: 0.8886" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3474 - accuracy: 0.8838 - val_loss: 0.3798 - val_accuracy: 0.8816" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3396 - accuracy: 0.8878 - val_loss: 0.3807 - val_accuracy: 0.8836" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3421 - accuracy: 0.8866 - val_loss: 0.4152 - val_accuracy: 0.8744" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3436 - accuracy: 0.8841 - val_loss: 0.3785 - val_accuracy: 0.8824" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3466 - accuracy: 0.8839 - val_loss: 0.4277 - val_accuracy: 0.8728" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3424 - accuracy: 0.8872 - val_loss: 0.3922 - val_accuracy: 0.8874" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3440 - accuracy: 0.8883 - val_loss: 0.3680 - val_accuracy: 0.8876" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3437 - accuracy: 0.8855 - val_loss: 0.3280 - val_accuracy: 0.8948" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3397 - accuracy: 0.8861 - val_loss: 0.3639 - val_accuracy: 0.8862" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3433 - accuracy: 0.8856 - val_loss: 0.3577 - val_accuracy: 0.8878" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3421 - accuracy: 0.8850 - val_loss: 0.3635 - val_accuracy: 0.8856" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3383 - accuracy: 0.8874 - val_loss: 0.3512 - val_accuracy: 0.8938" \
       "1407/1407 [==============================] - 114s 81ms/step - loss: 0.3439 - accuracy: 0.8855 - val_loss: 0.3429 - val_accuracy: 0.8908" \
       "1407/1407 [==============================] - 115s 81ms/step - loss: 0.3399 - accuracy: 0.8869 - val_loss: 0.3593 - val_accuracy: 0.8890"

def get_points(raw_vals):
     r = raw_vals.split("step")
     d = len(r)
     dict = {"train_loss": [],
             "train_acc": [],
             "val_loss": [],
             "val_acc": []
             }
     for i in range(1, d):
         aaa = r[i].split("-")
         tl = aaa[1][7:13]
         ta = aaa[2][11:17]
         vl = aaa[3][11:17]
         va = aaa[4][15:21]
         dict["train_loss"].append(float(tl))
         dict["train_acc"].append(float(ta))
         dict["val_loss"].append(float(vl))
         dict["val_acc"].append(float(va))
         #print(aaa, tl, ta, vl, va)
     return dict

my_list = [get_points(rec1), get_points(rec2), get_points(rec3), get_points(l1)]
# plotting the points
cols = ["r", "g", "b", "c"]
#for i in range(len(my_list)):
#     print(cols[i])
#     plt.plot(my_list[i]["val_acc"], color=cols[i])
start = 0
plt.plot(my_list[1]["val_acc"][start:], color="r", label="val_acc(no augm)")
plt.plot(my_list[0]["val_acc"][start:], color="g")
plt.plot(my_list[2]["val_acc"][start:], color="b", label="val_acc(augm)")
plt.plot(my_list[3]["val_acc"][start:], color="c")
plt.plot(my_list[1]["train_acc"][start:], color=[0.8, 0.2, 0.2], label="train_acc(no augm)")
plt.plot(my_list[2]["train_acc"][start:], color=[0.1, 0.5, 0.9], label="train_acc(augm)")
#plt.plot(dict["train_acc"])
#plt.plot(dict["val_acc"])
plt.axhline(y=max(my_list[2]["val_acc"]), color='r', linestyle='--')
plt.axhline(y=min(my_list[2]["val_acc"][100:]), color='b', linestyle='--')

# naming the x axis
plt.xlabel('epochs')
# naming the y axis
plt.ylabel('accuracy')

# giving a title to my graph
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
plt.title('Training CIFAR')

# function to show the plot
plt.show()