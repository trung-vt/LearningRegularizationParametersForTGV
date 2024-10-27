from datetime import datetime


class Trainer:

    def __init__(self, args):

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = torch.device(device)

        def get_formatted_date(format_str="%m_%d-%H_%M"):
            formatted_date = datetime.now().strftime(format_str)
            return formatted_date

        model_loader = ModelLoader(
            config_choice=args.config,
            is_training=True
        )

        if args.output_dir is None:
            model_loader.config["log"]["save_dir"] = \
                f"./tmp/mri_model_{get_formatted_date()}"
        else:
            model_loader.config["log"]["save_dir"] = args.output_dir
        print(f"Output directory: {model_loader.config['log']['save_dir']}")

        device = args.device
        if device is None:
            device = model_loader.device
        else:
            model_loader.config["device"] = device
        torch.set_default_device(device)
        print(f"Device choice: {device}")

        pdhg_net = model_loader.init_new_model()
        print(f"Regularization: {pdhg_net.pdhg_solver.pdhg_algorithm}")

        print(f"Data path: {model_loader.config['data']['data_path']}")
        training_data_loader = get_data_loader(
            data_config=model_loader.config["data"],
            action="train", dataset_type="dynamically_generated", device=device)
        validation_data_loader = get_data_loader(
            data_config=model_loader.config["data"],
            action="val", dataset_type="dynamically_generated", device=device)

        learning_rate = model_loader.config["train"]["learning_rate"]

        optim = torch.optim.AdamW(
            pdhg_net.parameters(),
            # lr=args.lr,
            # lr=1e-3,
            lr=learning_rate,
            # weight_decay=args.weight_decay
            # weight_decay=1e-5
            weight_decay=model_loader.config["train"]["weight_decay"]
        )
        sched = WarmupLR(torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                # (args.Nepochs - args.warmup) * len(training_data_loader),
                (model_loader.config["train"]["expected_num_epochs"] - 1) *
                len(training_data_loader),
                # eta_min=args.lr / 30, verbose=False
                # eta_min=1e-3 / 30,
                eta_min=learning_rate / 30,
                # verbose=False
            ),
            # init_lr=args.lr / 30,
            # init_lr=1e-3 / 30,
            init_lr=learning_rate / 30,
            # num_warmup=args.warmup * len(training_data_loader)
            # num_warmup=1 * len(training_data_loader)
            num_warmup=(
                model_loader.config["train"]["warmup"] *
                len(training_data_loader))
        )

        train_logger = Logger(
            action="train",
            config=model_loader.config,
            force_overwrite=False
        )
        train_logger.init_metrics_logging_options()
        train_logger.init_model_saving_options(log_config=model_loader.config["log"])

        val_logger = Logger(
            action="val",
            config=model_loader.config,
            force_overwrite=False
        )
        val_logger.init_metrics_logging_options()
        val_logger.init_model_saving_options(log_config=model_loader.config["log"])

        # Store config and other logs if specified.
        if args.logs_local:
            # Only need to log_config_local once.
            train_logger.log_config_local(pdhg_net=pdhg_net)

            train_logger.log_data_info(data_loader=training_data_loader)
            val_logger.log_data_info(data_loader=validation_data_loader)

        # Initialize WandB for logging if specified.
        if args.uses_wandb:
            train_logger.init_wandb()

        metrics_evaluator = ImageMetricsEvaluator(device=device)

        num_epochs = model_loader.config["train"]["num_epochs"]
        # for epoch in tqdm(range(args.Nepochs)):
        # for epoch in tqdm(range(1000)):
        for epoch in tqdm(range(num_epochs)):
            # gc.collect()
            train_logger.current_epoch = epoch
            pdhg_net.train(True)

            training_data_iterator = tqdm(training_data_loader)
            train_avg_metrics = perform_epoch(
                data_iterator=training_data_iterator,
                model=pdhg_net,
                logger=train_logger,
                is_training=True,
                metrics_evaluator=metrics_evaluator,
                learning_rate_scheduler=sched,
                optimizer=optim,
                sets_tqdm_postfix=True)

            pdhg_net.train(False)

            with torch.no_grad():
                torch.cuda.empty_cache()
                val_logger.current_epoch = epoch

                validation_data_iterator = tqdm(validation_data_loader)
                val_avg_metrics = perform_epoch(
                    data_iterator=validation_data_iterator,
                    model=pdhg_net,
                    logger=val_logger,
                    is_training=False,
                    metrics_evaluator=metrics_evaluator,
                    sets_tqdm_postfix=True)
                torch.cuda.empty_cache()

            if wandb.run is not None:
                wandb.log({"epoch": epoch+1})
            train_logger.log_metrics(
                stage="epoch", metrics=train_avg_metrics, iter_idx=None)
            val_logger.log_metrics(
                stage="epoch", metrics=val_avg_metrics, iter_idx=None)
            print(f"Epoch {epoch+1}:")
            print(
                f"TRAINING LOSS: {train_avg_metrics[0]}, " +
                f"TRAINING PSNR: {train_avg_metrics[1]:.2f}, " +
                f"TRAINING SSIM: {train_avg_metrics[2]:.4f}")
            print(
                f"VALIDATION LOSS: {val_avg_metrics[0]}, " +
                f"VALIDATION PSNR: {val_avg_metrics[1]:.2f}, " +
                f"VALIDATION SSIM: {val_avg_metrics[2]:.4f}")
            # torch.save(pdhg_net.state_dict(), f"./model_state_dict_{epoch}.pt")
            train_logger.save_model(pdhg_net=pdhg_net, idx=epoch, is_final=False)

        # Save as cpu for cross-device compatibility.
        pdhg_net.cpu()
        train_logger.save_model(pdhg_net=pdhg_net, idx=None, is_final=True)
        if args.savefile is None:
            args.savefile = f"./model_{get_formatted_date()}.pt"
        tosave = deepcopy(model_loader.config)
        tosave["state"] = pdhg_net.state_dict()
        torch.save(tosave, args.savefile)
        print(f"saved to {args.savefile}")
        # torch.save(pdhg_net.state_dict(), "./final_model_state_dict.pt")

        # if wandb.run is not None:
        #     wandb.log_model(
        #         final_model_path, name="final_model_state_dict")

        # if args.neptune:
        #     run.stop()
        if wandb.run is not None:
            wandb.finish()
