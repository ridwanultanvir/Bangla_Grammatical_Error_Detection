if eval_config.with_ground:
        for key in tokenized_train_dataset.keys():
            temp_offset_mapping = tokenized_train_dataset[key]["offset_mapping"]
            predictions = trainer.predict(tokenized_train_dataset[key])
            temp_untokenized_spans = untokenized_train_dataset[key]["spans"]
            # print(untokenized_train_dataset[key])

            preds = predictions.predictions
            preds = np.argmax(preds, axis=2)
            f1_scores = []
            edit_scores = []
            with open(
                # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
                os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
            ) as f:
                f.write(f"Model Name: {suffix}, Dataset: {key}\n")
            
            with open(
                # os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
                os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
            ) as f:
                for i, pred in tqdm(enumerate(preds), total=len(preds)):
                    # print(key,i)
                    ## Batch Wise
                    # print(len(prediction))
                    predicted_spans = []
                    for j, tokenwise_prediction in enumerate(
                        pred[: len(temp_offset_mapping[i])]
                    ):
                        if tokenwise_prediction == 1:
                            predicted_spans += list(
                                range(
                                    temp_offset_mapping[i][j][0],
                                    temp_offset_mapping[i][j][1],
                                )
                            )
                    if i == len(preds) - 1:
                        f.write(f"{i}\t{str(predicted_spans)}")
                    else:
                        f.write(f"{i}\t{str(predicted_spans)}\n")
                    
                    # Edit distance
                    ranges = get_ranges(predicted_spans)
                    # print("ranges: ", ranges)
                    # Get substring from span range
                    sentence = untokenized_train_dataset[key]["sentence"][i]
                    gt       = untokenized_train_dataset[key]["gt"][i]
                    output = ""
                    prev_s = 0
                    for i, span in enumerate(ranges):
                        if type(span) == int:
                            s = span
                            e = span + 1
                        else:
                            s = span.start
                            e = span.stop + 1
                        
                        output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
                        prev_s = e
                        
                    if prev_s < len(sentence):
                        output += sentence[prev_s:]
                    
                    output = replace_fixed(output)
                    output = replace_end(output)
                    
                    # print("output: ", output)
                    # gt = untokenized_train_dataset[key]["gt"][i]
                    # save output and gt to file
                    # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
                    #     outfile.write(gt + "\n")
                    #     # outfile.write(sentence + "\n")
                    #     outfile.write(output + "\n")
                    #     outfile.write("\n")

                    edit = edit_distance(output, gt)

                    edit_scores.append(edit)

                    f1_scores.append(
                        f1(
                            predicted_spans,
                            eval(temp_untokenized_spans[i]),
                        )
                    )
            with open(
                # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
                os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
            ) as f:
                f.write(str(np.mean(f1_scores)))
                f.write(" ")
                f.write(str(np.mean(edit_scores)))
                f.write("\n")