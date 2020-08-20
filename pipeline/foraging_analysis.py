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
class BlockRewardFraction(dj.Computed):
    definition = """ # Block reward fraction without bias check
    -> experiment.SessionBlock
    ---
    block_length: smallint #
    block_fraction: float
    first_tertile_fraction: float
    second_tertile_fraction: float
    third_tertile_fraction: float
    """

    class WaterPortFraction(dj.Part):
        definition = """
        -> master
        -> experiment.WaterPort
        ---
        wp_block_fraction=null: float
        wp_first_tertile_fraction=null: float
        wp_second_tertile_fraction=null: float
        wp_third_tertile_fraction=null: float
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
        Only process the blocks with:
         1. trial-count > minimum_trial_per_block
         2. is_real_training only
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
                'wp_block_fraction': rw.sum() / block_rw.sum() if block_rw.sum() else np.nan,
                'wp_first_tertile_fraction': (rw[:tertilelength].sum() / block_rw[:tertilelength].sum()
                                              if block_rw[:tertilelength].sum() else np.nan),
                'wp_second_tertile_fraction': (rw[tertilelength:2 * tertilelength].sum() / block_rw[tertilelength:2 * tertilelength].sum()
                                               if block_rw[tertilelength:2 * tertilelength].sum() else np.nan),
                'wp_third_tertile_fraction': (rw[-tertilelength:].sum() / block_rw[-tertilelength:].sum()
                                              if block_rw[-tertilelength:].sum() else np.nan)}

            wp_r_frac[water_port]['wp_incremental_fraction'] = np.full(len(self._window_ends), np.nan)
            for i, (t_start, t_end) in enumerate(zip(self._window_starts, self._window_ends)):
                if trialnum >= t_end and block_rw[t_start:t_end].sum() > 0:
                    wp_r_frac[water_port]['wp_incremental_fraction'][i] = (rw[t_start:t_end].sum()
                                                                           / block_rw[t_start:t_end].sum())

        self.WaterPortFraction.insert([{**key, 'water_port': wp, **wp_data} for wp, wp_data in wp_r_frac.items()])


@schema
class BlockEfficiency(dj.Computed):  # bias check excluded
    definition = """
    -> BlockRewardFraction
    ---
    block_effi_one_p_reward: float                       # denominator = max of the reward assigned probability (no baiting)
    block_effi_one_p_reward_first_tertile: float         # first tertile
    block_effi_one_p_reward_second_tertile: float        # second tertile
    block_effi_one_p_reward_third_tertile: float         # third tertile
    block_effi_sum_p_reward: float                       # denominator = sum of the reward assigned probability (no baiting)
    block_effi_sum_p_reward_first_tertile: float         # first tertile
    block_effi_sum_p_reward_second_tertile: float        # second tertile
    block_effi_sum_p_reward_third_tertile: float         # third tertile
    block_effi_one_a_reward=null: float                  # denominator = max of the reward assigned probability + baiting)
    block_effi_one_a_reward_first_tertile=null: float    # first tertile
    block_effi_one_a_reward_second_tertile=null: float   # second tertile
    block_effi_one_a_reward_third_tertile=null: float    # third tertile
    block_effi_sum_a_reward=null: float                  # denominator = sum of the reward assigned probability + baiting)
    block_effi_sum_a_reward_first_tertile=null: float    # first tertile
    block_effi_sum_a_reward_second_tertile=null: float   # second tertile
    block_effi_sum_a_reward_third_tertile=null: float    # third tertile
    block_ideal_phat_greedy=null: float                  # denominator = Ideal-pHat-greedy
    regret_ideal_phat_greedy=null: float                 # Ideal-pHat-greedy - reward collected
    """

    def make(self, key):
        water_ports, rewards = (experiment.SessionBlock.WaterPortRewardProbability & key).fetch(
            'water_port', 'reward_probability')
        rewards = rewards.astype(float)
        max_prob_reward = np.nanmax(rewards)
        sum_prob_reward = np.nansum(rewards)

        q_block_trial = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key
        tertilelength = int(np.floor(len(q_block_trial) / 3))

        reward_available = pd.DataFrame({wp: (q_block_trial * experiment.TrialAvailableReward
                                              & {'water_port': wp}).fetch('reward_available', order_by='trial')
                                         for wp in water_ports})
        max_reward_available = reward_available.max(axis=1)
        max_reward_available_first = max_reward_available[:tertilelength]
        max_reward_available_second = max_reward_available[tertilelength:2 * tertilelength]
        max_reward_available_third = max_reward_available[-tertilelength:]

        sum_reward_available = reward_available.sum(axis=1)
        sum_reward_available_first = sum_reward_available[:tertilelength]
        sum_reward_available_second = sum_reward_available[tertilelength:2 * tertilelength]
        sum_reward_available_third = sum_reward_available[-tertilelength:]

        block_reward_fraction = (BlockRewardFraction & key).fetch1()

        block_efficiency_data = dict(
            block_effi_one_p_reward=block_reward_fraction['block_fraction'] / max_prob_reward,
            block_effi_one_p_reward_first_tertile=block_reward_fraction['first_tertile_fraction'] / max_prob_reward,
            block_effi_one_p_reward_second_tertile=block_reward_fraction['second_tertile_fraction'] / max_prob_reward,
            block_effi_one_p_reward_third_tertile=block_reward_fraction['third_tertile_fraction'] / max_prob_reward,
            block_effi_sum_p_reward=block_reward_fraction['block_fraction'] / sum_prob_reward,
            block_effi_sum_p_reward_first_tertile=block_reward_fraction['first_tertile_fraction'] / sum_prob_reward,
            block_effi_sum_p_reward_second_tertile=block_reward_fraction['second_tertile_fraction'] / sum_prob_reward,
            block_effi_sum_p_reward_third_tertile=block_reward_fraction['third_tertile_fraction'] / sum_prob_reward,
            block_effi_one_a_reward=(block_reward_fraction['block_fraction'] / max_reward_available.mean()
                                     if max_reward_available.mean() else np.nan),
            block_effi_one_a_reward_first_tertile=(block_reward_fraction['first_tertile_fraction'] / max_reward_available_first.mean()
                                                   if max_reward_available_first.mean() else np.nan),
            block_effi_one_a_reward_second_tertile=(block_reward_fraction['second_tertile_fraction'] / max_reward_available_second.mean()
                                                    if max_reward_available_second.mean() else np.nan),
            block_effi_one_a_reward_third_tertile=(block_reward_fraction['third_tertile_fraction'] / max_reward_available_third.mean()
                                                   if max_reward_available_third.mean() else np.nan),
            block_effi_sum_a_reward=(block_reward_fraction['block_fraction'] / sum_reward_available.mean()
                                     if sum_reward_available.mean() else np.nan),
            block_effi_sum_a_reward_first_tertile=(block_reward_fraction['first_tertile_fraction'] / sum_reward_available_first.mean()
                                                   if sum_reward_available_first.mean() else np.nan),
            block_effi_sum_a_reward_second_tertile=(block_reward_fraction['second_tertile_fraction'] / sum_reward_available_second.mean()
                                                    if sum_reward_available_second.mean() else np.nan),
            block_effi_sum_a_reward_third_tertile=(block_reward_fraction['third_tertile_fraction'] / sum_reward_available_third.mean()
                                                   if sum_reward_available_third.mean() else np.nan))

        # Ideal-pHat-greedy  - only for blocks containing "left" and "right" port only
        if not len(np.setdiff1d(['right', 'left'], water_ports)):
            p1 = np.nanmax(rewards)
            p0 = np.nanmin(rewards)
            if p0 != 0:
                m_star_greedy = math.floor(math.log(1 - p1) / math.log(1 - p0))
                p_star_greedy = p1 + (1 - (1 - p0) ** (m_star_greedy + 1) - p1 ** 2) / (m_star_greedy + 1)

                block_efficiency_data.update(
                    block_ideal_phat_greedy=block_reward_fraction['block_fraction'] / p_star_greedy,
                    regret_ideal_phat_greedy=p_star_greedy - block_reward_fraction['block_fraction'])

        self.insert1({**key, **block_efficiency_data})
