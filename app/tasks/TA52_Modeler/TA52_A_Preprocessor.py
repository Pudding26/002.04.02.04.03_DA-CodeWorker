from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob



class TA52_A_Preprocessor:

    @staticmethod
    def run(job: ModelerJob) -> None:
        cfg = job.input.preProcessing_instructions
        X = job.attrs.raw_data

        match cfg.method:
            case "scaling":
                match cfg.scaling.submethod:
                    case "standardization":
                        match cfg.scaling.standardization.subsubmethod:
                            case "MinMaxScaler":
                                X_scaled = TA52_A_Preprocessor._minmax_scaler(X, cfg)
                            case "StandardScaler":
                                X_scaled = TA52_A_Preprocessor._standard_scaler(X, cfg)
                            case _:
                                raise ValueError(f"Unsupported standardization.subsubmethod: {cfg.scaling.standardization.subsubmethod}")
                    case _:
                        raise ValueError(f"Unsupported scaling.submethod: {cfg.scaling.submethod}")

            case "resampling":
                # Placeholder logic
                X_scaled = X

            case "none":
                job.attrs["preProcessing_status"] = "SKIP"
                return

            case _:
                raise ValueError(f"Unsupported preProcessing method: {cfg.method}")

        job.attrs.preProcessed_data = X_scaled
        job.attrs["preProcessing_status"] = "OK"



    @staticmethod
    def _standard_scaler(X, cfg):
        from cuml.preprocessing import StandardScaler
        p = cfg.scaling.standardization.StandardScaler
        scaler = StandardScaler(with_mean=p.with_mean, with_std=p.with_std)
        return scaler.fit_transform(X)

    @staticmethod
    def _minmax_scaler(X, cfg):
        from cuml.preprocessing import MinMaxScaler
        p = cfg.scaling.standardization.MinMaxScaler
        scaler = MinMaxScaler(feature_range=(p.min, p.max), clip=p.clip)
        return scaler.fit_transform(X)

