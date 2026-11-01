from src import utility
import torchvision.transforms as transforms

def test_add_transform_one():
    # Expect
    v_expect_transforms = [
        transforms.RandomAffine,
        transforms.ColorJitter,
        transforms.Normalize
    ]

    # Base
    v_base_tf = [
        transforms.RandomAffine(0.1),
        transforms.ColorJitter(0.1)
    ]

    v_add_tf = [
        transforms.Normalize(
            mean=0.7,
            std=0.5
        )
    ]


    v_result = utility.add_transform(v_base_tf, v_add_tf)
    assert [type(tf) for tf in v_result.transforms] == v_expect_transforms

def test_add_transform_many():
    # Expect
    v_expect_transforms = [
        transforms.RandomAffine,
        transforms.ColorJitter,
        transforms.ToTensor,
        transforms.Normalize
    ]

    # Base
    v_base_tf = [
        transforms.RandomAffine(0.1),
        transforms.ColorJitter(0.1)
    ]

    # Add two
    v_add_tf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.7,
            std=0.5
        ),
    ]


    v_result = utility.add_transform(v_base_tf, v_add_tf)
    assert [type(tf) for tf in v_result.transforms] == v_expect_transforms


if __name__ == '__main__':
    # test_add_transform()
    pass