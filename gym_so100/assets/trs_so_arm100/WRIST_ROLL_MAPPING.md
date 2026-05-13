# Wrist_Roll 关节值 ↔ 实际旋转 的映射调整

## 方案 1：反转旋转方向（轴取反）

若你希望「数值增大时」旋转方向相反，把 `axis` 取反：

```xml
<default class="Wrist_Roll">
  <joint axis="0 -1 0" range="-2.79 2.79" />   <!-- 原 axis="0 1 0" -->
</default>
```

或在 joint 上直接写：`<joint name="Wrist_Roll" class="Wrist_Roll" axis="0 -1 0" />`

---

## 方案 2：扩大关节范围

若 2.79 rad (≈160°) 仍不够，可增大 range（注意避免自碰撞）：

```xml
<joint axis="0 1 0" range="-3.14 3.14" />   <!-- ±180° -->
```

---

## 方案 3：零位偏移（改父 body 的 euler）

在 `so_arm100_puzzle.xml` 第 106 行，Fixed_Jaw 的 `euler="0 1.57079 0"` 决定 joint=0 时的姿态。

- 增大 Y 分量：joint=0 时末端多转一点
- 减小 Y 分量：joint=0 时末端少转一点

例如加 0.5 rad 偏移：
```xml
<body name="Fixed_Jaw" pos="0 -0.0601 0" euler="0 2.07079 0">  <!-- 原 1.57079 → 2.07079 -->
```

---

## 方案 4：代码中做缩放/偏移（最灵活）

在控制循环里，对 Wrist_Roll 的 ctrl 做映射，例如：

```python
# Wrist_Roll 是第 5 个 actuator (索引 4)
WRIST_ROLL_IDX = 4
SCALE = 1.2   # 你写 1.0 时，实际转 1.2 rad
OFFSET = 0.3  # 零位偏移

# 在 mj_step 之前：
desired = data.ctrl[WRIST_ROLL_IDX]  # 你期望的角度
data.ctrl[WRIST_ROLL_IDX] = desired * SCALE + OFFSET
```

这样可以在不改 XML 的前提下，自定义「期望角度 → 实际关节角」的映射。
