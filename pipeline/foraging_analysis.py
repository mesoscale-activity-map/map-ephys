import datajoint as dj
import numpy as np
import pandas as pd
import math

from pipeline import get_schema_name, experiment

schema = dj.schema(get_schema_name('foraging_analysis'))

block_reward_ratio_increment_step = 10
block_reward_ratio_increment_window = 20
block_reward_ratio_increment_max = 200
bootstrapnum = 100
minimum_trial_per_block = 30


@schema
class BlockRewardFractionNoBiasCheck(dj.Computed):  # without bias check - Naming is WRONG it's FRACTION
    definition = """
    -> experiment.SessionBlock
    ---
    block_length = null: smallint #
    block_fraction: float
    first_tertile_fraction: float
    second_tertile_fraction: float
    third_tertile_fraction: float
    """

    class WaterPortRewardFraction(dj.Part):
        definition = """
        -> master
        -> experiment.WaterPort
        ---
        wp_block_fraction: float
        wp_first_tertile_fraction: float
        wp_second_tertile_fraction: float
        wp_third_tertile_fraction: float
        wp_incremental_fraction: longblob
        """

    _window_starts = (np.arange(block_reward_ratio_increment_window / 2,
                                block_reward_ratio_increment_max,
                                block_reward_ratio_increment_step, dtype=int)
                      - int(round(block_reward_ratio_increment_window / 2)))
    _window_ends = (np.arange(block_reward_ratio_increment_window / 2,
                              block_reward_ratio_increment_max,
                              block_reward_ratio_increment_step, dtype=int)
                    + int(round(block_reward_ratio_increment_window / 2)))

    @property
    def key_source(self):
        """
        process block with trial-count > minimum_trial_per_block and is_real_training only
        """
        # trial-count > minimum_trial_per_block
        ks_tr_count = experiment.SessionBlock.aggr(experiment.SessionBlock.BlockTrial, tr_count='count(*)') & 'tr_count > {}'.format(minimum_trial_per_block)
        # is_real_training only
        ks_real = ks_tr_count - (experiment.SessionBlock.WaterPortRewardProbability & 'reward_probability >= 1')
        return ks_real

    def make(self, key):
        # To skip bias check trial 04/02/20 NW

        q_block_trial = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key

        block_rw = q_block_trial.proj(reward='outcome = "hit"').fetch('reward', order_by='trial')

        trialnum = len(block_rw)
        tertilelength = int(np.floor(trialnum / 3))

        # ---- whole-block fraction ----
        block_reward_fraction = dict(
            block_length=trialnum,
            block_fraction=block_rw.mean(),
            first_tertile_fraction=block_rw[:tertilelength].mean(),
            second_tertile_fraction=block_rw[tertilelength:2 * tertilelength].mean(),
            third_tertile_fraction=block_rw[-tertilelength:].mean())

        self.insert1({**key, **block_reward_fraction})

        # ---- water-port fraction ----
        wp_r_frac = {}
        for water_port in experiment.WaterPort.fetch('water_port'):
            rw = (experiment.WaterPortChoice * q_block_trial).proj(
                reward='water_port = "{}" AND outcome = "hit"'.format(water_port)).fetch('reward', order_by='trial')
            wp_r_frac[water_port] = {
                'wp_block_fraction': rw.sum() / block_rw.sum(),
                'wp_first_tertile_fraction': rw[:tertilelength].sum() / block_rw[:tertilelength].sum(),
                'wp_second_tertile_fraction': rw[tertilelength:2 * tertilelength].sum() / block_rw[tertilelength:2 * tertilelength].sum(),
                'wp_third_tertile_fraction': rw[-tertilelength:].sum() / block_rw[-tertilelength:].sum()}

            wp_r_frac[water_port]['wp_incremental_fraction'] = np.full(len(self._window_ends), np.nan)
            for i, (t_start, t_end) in enumerate(zip(self._window_starts, self._window_ends)):
                if trialnum >= t_end and block_rw[t_start:t_end].sum() > 0:
                    wp_r_frac[water_port]['wp_incremental_fraction'][i] = (rw[t_start:t_end].sum()
                                                                           / block_rw[t_start:t_end].sum())

        self.WaterPortRewardFraction.insert([{**key, 'water_port': wp, **wp_data} for wp, wp_data in wp_r_frac.items()])


@schema
class BlockEfficiency(dj.Computed):  # bias check excluded
    definition = """
    -> experiment.SessionBlock
    ---
    block_effi_one_p_reward =  null: float # denominator = max of the reward assigned probability (no baiting)
    block_effi_one_p_reward_first_tertile =  null: float # first tertile
    block_effi_one_p_reward_second_tertile =  null: float # second tertile
    block_effi_one_p_reward_third_tertile =  null: float # third tertile
    block_effi_sum_p_reward =  null: float # denominator = sum of the reward assigned probability (no baiting)
    block_effi_sum_p_reward_first_tertile =  null: float # first tertile
    block_effi_sum_p_reward_second_tertile =  null: decimal(8,4) # second tertile
    block_effi_sum_p_reward_third_tertile =  null: decimal(8,4) # third tertile
    block_effi_one_a_reward =  null: decimal(8,4) # denominator = max of the reward assigned probability + baiting)
    block_effi_one_a_reward_first_tertile =  null: decimal(8,4) # first tertile
    block_effi_one_a_reward_second_tertile =  null: decimal(8,4) # second tertile
    block_effi_one_a_reward_third_tertile =  null: decimal(8,4) # third tertile
    block_effi_sum_a_reward =  null: decimal(8,4) # denominator = sum of the reward assigned probability + baiting)
    block_effi_sum_a_reward_first_tertile =  null: decimal(8,4) # first tertile
    block_effi_sum_a_reward_second_tertile =  null: decimal(8,4) # second tertile
    block_effi_sum_a_reward_third_tertile =  null: decimal(8,4) # third tertile
    block_ideal_phat_greedy = null: decimal(8,4) # denominator = Ideal-pHat-greedy
    regret_ideal_phat_greedy = null: decimal(8,4) # Ideal-pHat-greedy - reward collected
    """

    def make(self, key):
        # key = {'subject_id': 447921, 'session': 12, 'block': 17}
        keytoinsert = key
        print(keytoinsert)
        df_choices = pd.DataFrame(experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key)
        realtraining = (df_choices['p_reward_left'] < 1) & (df_choices['p_reward_right'] < 1) & (
                    (df_choices['p_reward_middle'] < 1) | df_choices['p_reward_middle'].isnull())

        if realtraining[0] and len(df_choices) > minimum_trial_per_block:
            p_reward_left, p_reward_right, p_reward_middle = (experiment.SessionBlock() & keytoinsert).fetch(
                'p_reward_left', 'p_reward_right', 'p_reward_middle')
            p_reward_left = p_reward_left.astype(float)
            p_reward_right = p_reward_right.astype(float)
            p_reward_middle = p_reward_middle.astype(float)
            max_prob_reward = np.nanmax([p_reward_left, p_reward_right, p_reward_middle])
            sum_prob_reward = np.nansum([p_reward_left, p_reward_right, p_reward_middle])

            reward_available = pd.DataFrame((experiment.BehaviorTrial() * experiment.SessionBlock() * experiment.TrialAvailableReward() & keytoinsert))
            reward_available_left = reward_available['trial_available_reward_left']
            reward_available_right = reward_available['trial_available_reward_right']
            reward_available_middle = reward_available['trial_available_reward_middle']

            tertilelength = int(np.floor(len(reward_available_left) / 3))
            if reward_available_middle[0]:
                max_reward_available = reward_available[["trial_available_reward_left", "trial_available_reward_right",
                                                         "trial_available_reward_middle"]].max(axis = 1)
                max_reward_available_first = max_reward_available[:tertilelength]
                max_reward_available_second = max_reward_available[tertilelength:2 * tertilelength]
                max_reward_available_third = max_reward_available[-tertilelength:]

                sum_reward_available = reward_available_left + reward_available_right + reward_available_middle
                sum_reward_available_first = sum_reward_available[:tertilelength]
                sum_reward_available_second = sum_reward_available[tertilelength:2 * tertilelength]
                sum_reward_available_third = sum_reward_available[-tertilelength:]
            else:
                max_reward_available = reward_available[
                    ["trial_available_reward_left", "trial_available_reward_right"]].max(axis = 1)
                max_reward_available_first = max_reward_available[:tertilelength]
                max_reward_available_second = max_reward_available[tertilelength:2 * tertilelength]
                max_reward_available_third = max_reward_available[-tertilelength:]

                sum_reward_available = reward_available_left + reward_available_right
                sum_reward_available_first = sum_reward_available[:tertilelength]
                sum_reward_available_second = sum_reward_available[tertilelength:2 * tertilelength]
                sum_reward_available_third = sum_reward_available[-tertilelength:]

                p1 = np.nanmax([p_reward_left, p_reward_right])
                p0 = np.nanmin([p_reward_left, p_reward_right])
                if p0 != 0:
                    m_star_greedy = math.floor(math.log(1 - p1) / math.log(1 - p0))
                    p_star_greedy = p1 + (1 - (1 - p0) ** (m_star_greedy + 1) - p1 ** 2) / (m_star_greedy + 1)

            BlockRewardRatio = pd.DataFrame(BlockRewardRatioNoBiasCheck & key)

            keytoinsert['block_effi_one_p_reward'] = float(BlockRewardRatio['block_reward_ratio']) / max_prob_reward
            try:
                keytoinsert['block_effi_one_p_reward_first_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_first_tertile']) / max_prob_reward
            except:
                pass
            try:
                keytoinsert['block_effi_one_p_reward_second_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_second_tertile']) / max_prob_reward
            except:
                pass
            try:
                keytoinsert['block_effi_one_p_reward_third_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_third_tertile']) / max_prob_reward
            except:
                pass
            try:
                keytoinsert['block_effi_sum_p_reward'] = float(BlockRewardRatio['block_reward_ratio']) / sum_prob_reward
            except:
                pass
            try:
                keytoinsert['block_effi_sum_p_reward_first_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_first_tertile']) / sum_prob_reward
            except:
                pass
            try:
                keytoinsert['block_effi_sum_p_reward_second_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_second_tertile']) / sum_prob_reward
            except:
                pass
            try:
                keytoinsert['block_effi_sum_p_reward_third_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_third_tertile']) / sum_prob_reward
            except:
                pass
            if max_reward_available.mean() != 0 and BlockRewardRatio['block_reward_ratio'][0] is not None:
                keytoinsert['block_effi_one_a_reward'] = float(
                    BlockRewardRatio['block_reward_ratio']) / max_reward_available.mean()
            else:
                keytoinsert['block_effi_one_a_reward'] = np.nan
            if max_reward_available_first.mean() != 0 and BlockRewardRatio['block_reward_ratio_first_tertile'][
                0] is not None:
                keytoinsert['block_effi_one_a_reward_first_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_first_tertile']) / max_reward_available_first.mean()
            else:
                keytoinsert['block_effi_one_a_reward_first_tertile'] = np.nan
            if max_reward_available_second.mean() != 0 and BlockRewardRatio['block_reward_ratio_second_tertile'][
                0] is not None:
                keytoinsert['block_effi_one_a_reward_second_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_second_tertile']) / max_reward_available_second.mean()
            else:
                keytoinsert['block_effi_one_a_reward_second_tertile'] = np.nan
            if max_reward_available_third.mean() != 0 and BlockRewardRatio['block_reward_ratio_third_tertile'][
                0] is not None:
                keytoinsert['block_effi_one_a_reward_third_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_third_tertile']) / max_reward_available_third.mean()
            else:
                keytoinsert['block_effi_one_a_reward_third_tertile'] = np.nan

            if sum_reward_available.mean() != 0 and BlockRewardRatio['block_reward_ratio'][0] is not None:
                keytoinsert['block_effi_sum_a_reward'] = float(
                    BlockRewardRatio['block_reward_ratio']) / sum_reward_available.mean()
            else:
                keytoinsert['block_effi_sum_a_reward'] = np.nan
            if sum_reward_available_first.mean() != 0 and BlockRewardRatio['block_reward_ratio_first_tertile'][
                0] is not None:
                keytoinsert['block_effi_sum_a_reward_first_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_first_tertile']) / sum_reward_available_first.mean()
            else:
                keytoinsert['block_effi_sum_a_reward_first_tertile'] = np.nan
            if sum_reward_available_second.mean() != 0 and BlockRewardRatio['block_reward_ratio_second_tertile'][
                0] is not None:
                keytoinsert['block_effi_sum_a_reward_second_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_second_tertile']) / sum_reward_available_second.mean()
            else:
                keytoinsert['block_effi_sum_a_reward_second_tertile'] = np.nan
            if sum_reward_available_third.mean() != 0 and BlockRewardRatio['block_reward_ratio_third_tertile'][
                0] is not None:
                keytoinsert['block_effi_sum_a_reward_third_tertile'] = float(
                    BlockRewardRatio['block_reward_ratio_third_tertile']) / sum_reward_available_third.mean()
            else:
                keytoinsert['block_effi_sum_a_reward_third_tertile'] = np.nan
            try:
                keytoinsert['block_ideal_phat_greedy'] = float(BlockRewardRatio['block_reward_ratio']) / p_star_greedy
            except:
                pass
            try:
                keytoinsert['regret_ideal_phat_greedy'] = p_star_greedy - float(BlockRewardRatio['block_reward_ratio'])
            except:
                pass

        self.insert1(keytoinsert, skip_duplicates = True)

