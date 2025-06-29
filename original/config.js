// this seems to fascilitate a little bit of exploring, rather than just facing a wall waiting to die.
// seems to cause disco head
const rewardConfigs={
    default: {
        baseReward: 1,
        hitShotReward: 1,
        biteReward: -1,
        hitHumanReward: -1,
        missedShotReward: -.125,
        bumpWallReward: -.55,
        bumpScreenReward: -.75,
        bumpHumanReward: -.65,
        blockedVisionHuman: 0,
        blockedVisionWall: -.1,
        farVisionReward: 0,
        zombieProximityReward: 0
    },
    explore: {
        baseReward: 1,
        hitShotReward: 1,
        biteReward: -1,
        hitHumanReward: -1,
        missedShotReward: -.1,
        bumpWallReward: -.5,
        bumpScreenReward: -.5,
        bumpHumanReward: -.65,
        blockedVisionHuman: 0,
        blockedVisionWall: -.5,
        farVisionReward: 0,
        zombieProximityReward: -.1
    },
    attack: {
        baseReward: 1,
        hitShotReward: 1.5,
        biteReward: -1,
        hitHumanReward: -1,
        missedShotReward: -.2,
        bumpWallReward: -.55,
        bumpScreenReward: -.75,
        bumpHumanReward: -.65,
        blockedVisionHuman: -.5,
        blockedVisionWall: -.2,
        farVisionReward: .2,
        zombieProximityReward: -.1
    }
}
